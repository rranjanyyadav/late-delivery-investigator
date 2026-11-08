"""Loader, views and data-quality checks on the tiny dataset."""
import duckdb

from ldi.loader import CHECKS, run_quality_checks


def test_tiny_build_has_no_failed_checks(con):
    statuses = dict(con.execute("SELECT check_name, status FROM dq_results").fetchall())
    assert "FAIL" not in statuses.values()
    assert statuses["products with no category"] == "WARN"   # p2 has no category


def test_zip_codes_keep_leading_zeros(con):
    assert con.execute("SELECT customer_zip_code_prefix FROM customers "
                       "WHERE customer_id = 'c1'").fetchall()[0][0] == "01037"


def test_view_grain(con):
    assert con.execute("SELECT count(*) FROM order_delivery").fetchall()[0][0] == 4   # one per order
    assert con.execute("SELECT count(*) FROM orders_enriched").fetchall()[0][0] == 5  # one per item


def test_late_is_judged_by_day_not_by_hour(con):
    # o1 arrived at 09:00 on its estimated day (estimate is 00:00): on time.
    row = con.execute("SELECT is_late, delivery_delay_days FROM order_delivery "
                      "WHERE order_id = 'o1'").fetchall()[0]
    assert row == (False, 0)


def test_undelivered_orders_are_not_judged(con):
    row = con.execute("SELECT is_delivered, is_late FROM order_delivery "
                      "WHERE order_id = 'o3'").fetchall()[0]
    assert row == (False, None)


def test_checks_catch_injected_problems():
    con = duckdb.connect()  # in-memory: build bad data by hand
    con.execute("""CREATE TABLE orders AS SELECT * FROM (VALUES
        ('o1', 'delivered', TIMESTAMP '2018-07-02', TIMESTAMP '2018-07-01'),  -- delivered before purchase
        ('o1', 'delivered', TIMESTAMP '2018-07-02', TIMESTAMP '2018-07-05')   -- duplicate id
    ) t(order_id, order_status, order_purchase_timestamp, order_delivered_customer_date)""")
    wanted = {"duplicate orders.order_id", "delivered to customer before purchase"}
    results = {r.name: r for r in run_quality_checks(con, [c for c in CHECKS if c.name in wanted])}
    assert results["duplicate orders.order_id"].status == "FAIL"        # critical
    assert results["delivered to customer before purchase"].status == "WARN"
    assert results["delivered to customer before purchase"].bad_rows == 1
