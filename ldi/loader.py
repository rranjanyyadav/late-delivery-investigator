"""Module 1: load the Olist CSVs into DuckDB and run data-quality checks.

Usage:
    python -m ldi.loader            # build data/olist.duckdb and print a summary

The build is idempotent: it deletes the old DB file and rebuilds it from the
raw CSVs, so the database is always a pure function of data/raw/.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

from ldi.config import DB_PATH, RAW_DIR

# CSV file name -> table name in DuckDB.
TABLES = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_customers_dataset.csv": "customers",
    "olist_sellers_dataset.csv": "sellers",
    "olist_products_dataset.csv": "products",
    "olist_order_payments_dataset.csv": "payments",
    "olist_order_reviews_dataset.csv": "reviews",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "category_translation",
}

# The type sniffer would read zip prefixes like "01037" as integers and drop
# the leading zero, so we force them to text.
TYPE_OVERRIDES = {
    "customers": {"customer_zip_code_prefix": "VARCHAR"},
    "sellers": {"seller_zip_code_prefix": "VARCHAR"},
    "geolocation": {"geolocation_zip_code_prefix": "VARCHAR"},
}


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    """Run a query that returns one row with one column and return that value."""
    # fetchall() reads the result to the end, which DuckDB 2.x requires
    # before the same connection can run another statement.
    return con.execute(sql).fetchall()[0][0]


def load_csvs(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> dict[str, int]:
    """Create one table per CSV. Returns {table_name: row_count}."""
    counts = {}
    for file_name, table in TABLES.items():
        path = raw_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")
        # read_csv sniffs the delimiter, header and column types; we only
        # override the columns the sniffer gets wrong (see TYPE_OVERRIDES).
        options = "header = true"
        overrides = TYPE_OVERRIDES.get(table)
        if overrides:
            struct = ", ".join(f"'{col}': '{typ}'" for col, typ in overrides.items())
            options += f", types = {{{struct}}}"
        csv_path = path.as_posix().replace("'", "''")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv('{csv_path}', {options})"
        )
        counts[table] = scalar(con, f"SELECT count(*) FROM {table}")
    return counts


@dataclass
class Check:
    name: str
    sql: str            # must return a single number: how many rows are bad
    critical: bool = False  # critical -> a non-zero count fails the build


@dataclass
class CheckResult:
    name: str
    bad_rows: int
    critical: bool

    @property
    def status(self) -> str:
        if self.bad_rows == 0:
            return "PASS"
        return "FAIL" if self.critical else "WARN"


def _dup(table: str, key: str) -> str:
    """SQL counting how many key values appear more than once."""
    return f"SELECT count(*) FROM (SELECT {key} FROM {table} GROUP BY {key} HAVING count(*) > 1)"


CHECKS = [
    # 1. Duplicate IDs. Orders and order lines are the grain of everything we
    #    compute, so duplicates there would double-count -> critical.
    Check("duplicate orders.order_id", _dup("orders", "order_id"), critical=True),
    Check("duplicate order_items (order_id, order_item_id)",
          _dup("order_items", "order_id, order_item_id"), critical=True),
    Check("duplicate customers.customer_id", _dup("customers", "customer_id"), critical=True),
    Check("duplicate sellers.seller_id", _dup("sellers", "seller_id"), critical=True),
    Check("duplicate products.product_id", _dup("products", "product_id"), critical=True),
    Check("duplicate reviews.review_id", _dup("reviews", "review_id")),

    # 2. Nulls in columns our metrics depend on.
    Check("delivered orders with no delivery date",
          "SELECT count(*) FROM orders WHERE order_status = 'delivered' "
          "AND order_delivered_customer_date IS NULL"),
    Check("orders with no estimated delivery date",
          "SELECT count(*) FROM orders WHERE order_estimated_delivery_date IS NULL",
          critical=True),
    Check("products with no category",
          "SELECT count(*) FROM products WHERE product_category_name IS NULL"),
    Check("categories with no English translation",
          "SELECT count(DISTINCT p.product_category_name) FROM products p "
          "LEFT JOIN category_translation t USING (product_category_name) "
          "WHERE p.product_category_name IS NOT NULL "
          "AND t.product_category_name_english IS NULL"),

    # 3. Impossible dates: things that cannot happen before the purchase.
    Check("delivered to customer before purchase",
          "SELECT count(*) FROM orders "
          "WHERE order_delivered_customer_date < order_purchase_timestamp"),
    Check("handed to carrier before purchase",
          "SELECT count(*) FROM orders "
          "WHERE order_delivered_carrier_date < order_purchase_timestamp"),
    Check("approved before purchase",
          "SELECT count(*) FROM orders WHERE order_approved_at < order_purchase_timestamp"),
    Check("delivered to customer before handed to carrier",
          "SELECT count(*) FROM orders "
          "WHERE order_delivered_customer_date < order_delivered_carrier_date"),

    # 4. Referential integrity (orphan foreign keys) -> joins would drop rows.
    Check("order_items with unknown order_id",
          "SELECT count(*) FROM order_items i "
          "WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = i.order_id)",
          critical=True),
    Check("order_items with unknown seller_id",
          "SELECT count(*) FROM order_items i "
          "WHERE NOT EXISTS (SELECT 1 FROM sellers s WHERE s.seller_id = i.seller_id)",
          critical=True),
    Check("order_items with unknown product_id",
          "SELECT count(*) FROM order_items i "
          "WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = i.product_id)",
          critical=True),
    Check("orders with unknown customer_id",
          "SELECT count(*) FROM orders o "
          "WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id)",
          critical=True),
    Check("orders with no order_items",
          "SELECT count(*) FROM orders o "
          "WHERE NOT EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)"),
]


def run_quality_checks(con: duckdb.DuckDBPyConnection, checks=CHECKS) -> list[CheckResult]:
    """Run every check and store the results in a dq_results table."""
    results = [
        CheckResult(c.name, scalar(con, c.sql), c.critical) for c in checks
    ]
    con.execute(
        "CREATE OR REPLACE TABLE dq_results "
        "(check_name VARCHAR, bad_rows BIGINT, critical BOOLEAN, status VARCHAR)"
    )
    con.executemany(
        "INSERT INTO dq_results VALUES (?, ?, ?, ?)",
        [(r.name, r.bad_rows, r.critical, r.status) for r in results],
    )
    return results


def print_summary(counts: dict[str, int], results: list[CheckResult]) -> None:
    print("\nTables loaded")
    print("-" * 40)
    for table, n in counts.items():
        print(f"  {table:<24}{n:>12,}")

    print("\nData-quality checks")
    print("-" * 72)
    for r in results:
        print(f"  [{r.status}] {r.name:<55}{r.bad_rows:>8,}")
    n_fail = sum(r.status == "FAIL" for r in results)
    n_warn = sum(r.status == "WARN" for r in results)
    print(f"\n  {len(results)} checks: {n_fail} FAIL, {n_warn} WARN\n")


def build(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR) -> list[CheckResult]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # always rebuild from scratch
    con = duckdb.connect(str(db_path))
    try:
        counts = load_csvs(con, raw_dir)
        results = run_quality_checks(con)
    finally:
        con.close()
    print_summary(counts, results)
    print(f"Database written to {db_path}")
    return results


if __name__ == "__main__":
    results = build()
    # A non-zero exit code lets a CI job or a scheduler notice a bad load.
    sys.exit(1 if any(r.status == "FAIL" for r in results) else 0)
