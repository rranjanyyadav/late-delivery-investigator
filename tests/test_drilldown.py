"""The drill-down's contributions must add up to the change it explains."""
import pytest

from ldi.drilldown import decompose
from ldi.semantic import SemanticError, SemanticLayer

LAYER = SemanticLayer()


def test_contributions_sum_to_the_change(con):
    # Late rate: Q2 0/1 (o4 early) -> Q3 1/2 (o2 late), change +0.5, all from RJ.
    d = decompose(con, LAYER, "late_delivery_rate", "state", "quarter", "2018-Q2", "2018-Q3")
    assert d.change == pytest.approx(0.5)
    assert d.table["contribution"].sum() == pytest.approx(d.change)
    assert d.residual == pytest.approx(0)
    top = d.table.iloc[0]
    assert top["segment"] == "RJ" and top["contribution"] == pytest.approx(0.5)


def test_rate_plus_mix_equals_contribution(con):
    d = decompose(con, LAYER, "late_delivery_rate", "state", "quarter", "2018-Q2", "2018-Q3")
    t = d.table
    assert (t["rate_effect"] + t["mix_effect"] - t["contribution"]).abs().max() == pytest.approx(0)


def test_on_time_rate_is_decomposed_through_the_late_rate(con):
    d = decompose(con, LAYER, "on_time_delivery_rate", "state", "quarter", "2018-Q2", "2018-Q3")
    assert d.metric == "late_delivery_rate"


def test_non_ratio_metric_is_refused(con):
    with pytest.raises(SemanticError, match="cannot be decomposed"):
        decompose(con, LAYER, "revenue", "state", "quarter", "2018-Q2", "2018-Q3")


def test_missing_period_is_reported(con):
    with pytest.raises(SemanticError, match="No data"):
        decompose(con, LAYER, "late_delivery_rate", "state", "quarter", "2018-Q2", "2019-Q1")
