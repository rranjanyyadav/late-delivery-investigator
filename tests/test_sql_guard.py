"""The SQL guard must block every non-SELECT, and the connection must be
read-only even for a statement that got past the guard."""
import duckdb
import pytest

from ldi.sql_guard import UnsafeSQLError, run_query, validate_sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders",
    "DELETE FROM orders",
    "UPDATE orders SET order_status = 'x'",
    "INSERT INTO orders VALUES (1)",
    "CREATE TABLE t AS SELECT 1",
    "SELECT 1; DROP TABLE orders",                       # second statement
    "SELECT * FROM read_csv('data/raw/olist_orders_dataset.csv')",  # file read
    "ATTACH 'other.duckdb'",
    "COPY orders TO 'out.csv'",
    "PRAGMA database_list",
    "/* looks harmless */ DELETE FROM orders",
    "WITH x AS (SELECT 1) DELETE FROM orders",
    "",
])
def test_guard_blocks_unsafe_sql(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select count(*) from order_delivery;",              # trailing ; is fine
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT * FROM orders_enriched WHERE category = 'drop; delete'",  # keywords inside a string
    "SELECT 1 -- ; drop table orders",                    # inside a comment
    "SELECT replace(category, '_', ' ') FROM orders_enriched",
])
def test_guard_allows_plain_selects(sql):
    assert validate_sql(sql)


def test_connection_is_read_only_even_without_the_guard(con):
    with pytest.raises(duckdb.Error):
        con.execute("CREATE TABLE hacked AS SELECT 1")


def test_connection_cannot_read_files(con, tiny_db):
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT * FROM read_csv('{tiny_db.as_posix()}')")


def test_row_limit_truncates(con):
    df, truncated = run_query(con, "SELECT * FROM range(10)", max_rows=3)
    assert len(df) == 3 and truncated
    df, truncated = run_query(con, "SELECT * FROM range(3)", max_rows=3)
    assert len(df) == 3 and not truncated
