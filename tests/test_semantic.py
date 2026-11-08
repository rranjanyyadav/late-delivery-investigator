"""Every metric in metrics.yaml must compile to valid SQL, and give the right
number on the tiny dataset whose answers we know by hand."""
import pytest

from ldi.semantic import SemanticError, SemanticLayer, run

LAYER = SemanticLayer()
Q3 = [{"field": "quarter", "op": "=", "value": "2018-Q3"}]


@pytest.mark.parametrize("metric", list(LAYER.metrics))
def test_every_metric_compiles_and_runs(con, metric):
    run(con, LAYER.compile(metric))


@pytest.mark.parametrize("metric,dimension", [
    (m.name, d) for m in LAYER.metrics.values() for d in m.dimensions
])
def test_every_allowed_dimension_runs(con, metric, dimension):
    run(con, LAYER.compile(metric, dimensions=[dimension]))


@pytest.mark.parametrize("metric,expected", [
    # 2 delivered orders in Q3: o1 on time, o2 late. Averaging over order LINES
    # would give 2/3, because o1 has two lines; the metric must say 0.5.
    ("on_time_delivery_rate", 0.5),
    ("late_delivery_rate", 0.5),
    ("delivered_orders", 2),
    ("late_order_count", 1),
    ("avg_delay_days", 2.5),        # o1: 0 days, o2: +5 days
    ("revenue", 265.0),             # o1 110 + 55, o2 100; canceled o3 excluded
    ("seller_late_shipping_rate", 1 / 3),  # o2's line shipped after its limit
])
def test_metric_values_on_known_data(con, metric, expected):
    value = run(con, LAYER.compile(metric, filters=Q3)).iloc[0, 0]
    assert value == pytest.approx(expected)


def test_disallowed_dimension_is_refused():
    # avg_delay_days is per order; grouping it by seller would weight orders by lines.
    with pytest.raises(SemanticError, match="not allowed"):
        LAYER.compile("avg_delay_days", dimensions=["seller"])


def test_unknown_metric_lists_the_real_ones():
    with pytest.raises(SemanticError, match="on_time_delivery_rate"):
        LAYER.compile("on_time_percentage")


def test_metrics_from_different_views_cannot_be_mixed():
    with pytest.raises(SemanticError, match="different views"):
        LAYER.compile(["on_time_delivery_rate", "avg_delay_days"])


def test_filter_values_are_parameters_not_sql(con):
    evil = "SP' OR 1=1 --"
    q = LAYER.compile("delivered_orders", filters=[{"field": "state", "op": "=", "value": evil}])
    assert evil not in q.sql and q.params == [evil]
    assert run(con, q).iloc[0, 0] == 0   # matched nothing, changed nothing
