"""Module 2: create the analytics views from the plain SQL files in sql/.

The loader calls apply_views() on every build. You can also re-apply the
views to an existing database (handy while editing the SQL) with:

    python -m ldi.views
"""
from pathlib import Path

import duckdb

from ldi.config import DB_PATH, SQL_DIR


def apply_views(con: duckdb.DuckDBPyConnection, sql_dir: Path = SQL_DIR) -> list[str]:
    """Run every .sql file in name order (01_, 02_, ...) so that a view can
    depend on the views defined before it. Returns the file names applied."""
    applied = []
    for path in sorted(sql_dir.glob("*.sql")):
        con.execute(path.read_text(encoding="utf-8"))
        applied.append(path.name)
    return applied


def print_sanity_summary(con: duckdb.DuckDBPyConnection) -> None:
    """A few headline numbers, so a human can eyeball whether the views make sense."""
    overall = con.execute("""
        SELECT count(*)                          AS delivered_orders,
               round(100 * avg(CASE WHEN is_late THEN 0 ELSE 1 END), 2) AS on_time_pct,
               round(avg(delivery_delay_days), 2) AS avg_delay_days,
               round(sum(order_revenue), 2)       AS revenue
        FROM order_delivery
        WHERE is_delivered
    """).fetchall()[0]
    print("\nDelivered orders: {:,} | on-time: {}% | avg delay: {} days | revenue: R$ {:,}"
          .format(*overall))

    print("\nWorst 5 customer states by late rate (states with >= 1,000 delivered orders)")
    rows = con.execute("""
        SELECT customer_state,
               count(*)                              AS delivered_orders,
               round(100 * avg(is_late::INT), 2)     AS late_pct
        FROM order_delivery
        WHERE is_delivered
        GROUP BY customer_state
        HAVING count(*) >= 1000
        ORDER BY late_pct DESC
        LIMIT 5
    """).fetchall()
    for state, n, pct in rows:
        print(f"  {state}  {n:>7,} orders  {pct:>6}% late")


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    try:
        print("Applied:", ", ".join(apply_views(con)))
        print_sanity_summary(con)
    finally:
        con.close()
