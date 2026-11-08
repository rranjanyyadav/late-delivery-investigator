"""Module 5: root-cause drill-down. "Why did metric X change between two periods?"

For a ratio metric R = numerator / denominator (e.g. late orders / delivered
orders), split the change R1 - R0 into one contribution per segment
(category, seller, state, ...):

    contribution_i = k1_i / N1 - k0_i / N0

    k_i = segment i's numerator (its late orders), N = overall denominator
    (all delivered orders), 0 = base period, 1 = compare period.

Because R = sum_i(k_i) / N, these contributions add up exactly to R1 - R0
(see "residual" below). Each contribution is further split into:

    rate_effect_i = avg(w0_i, w1_i) * (r1_i - r0_i)   the segment got better/worse
    mix_effect_i  = avg(r0_i, r1_i) * (w1_i - w0_i)   the segment's share of volume changed

    w_i = n_i / N (segment's share of volume), r_i = k_i / n_i (segment's own rate).

rate_effect + mix_effect = contribution, exactly (it is the symmetric
"shift-share" decomposition). Ranking by contribution, not by the segment's
own rate, means a seller with 2 orders that were both late cannot top the
list: contributions are weighted by volume automatically.

Run `python -m ldi.drilldown` for the São Paulo 2018-Q2 -> Q3 example.
"""
from dataclasses import dataclass

import duckdb
import pandas as pd

from ldi.semantic import SemanticError, SemanticLayer


@dataclass
class Drilldown:
    metric: str
    dimension: str
    period_field: str
    base: str
    compare: str
    base_value: float      # overall metric in the base period
    compare_value: float   # overall metric in the compare period
    table: pd.DataFrame    # one row per segment, biggest contributor first
    residual: float        # change not assigned to any segment (see explain())
    sql: str
    params: list
    base_volume: float = 0     # overall denominator N0 (e.g. delivered orders)
    compare_volume: float = 0  # overall denominator N1

    @property
    def change(self) -> float:
        return self.compare_value - self.base_value


# A rate close to 100% must be decomposed through its complement. Almost every
# order is on time, so a segment's on-time contribution k1/N1 - k0/N0 is roughly
# its change in volume SHARE (w1 - w0): in testing, bed_bath_table "explained"
# -1.09 pp of the SP drop, -1.01 pp of it pure mix, while its own on-time rate
# barely moved. The late rate counts the rare event, so its contributions
# point at segments that actually got worse. The two changes are exact mirrors.
DECOMPOSE_AS = {"on_time_delivery_rate": "late_delivery_rate"}


def decompose(con: duckdb.DuckDBPyConnection, layer: SemanticLayer, metric: str,
              dimension: str, period_field: str, base: str, compare: str,
              filters=()) -> Drilldown:
    metric = DECOMPOSE_AS.get(metric, metric)
    m = layer.get_metric(metric)
    if not m.decomposable:
        options = [x.name for x in layer.metrics.values() if x.decomposable]
        raise SemanticError(f"Metric '{metric}' cannot be decomposed. Use one of: {', '.join(options)}.")
    period_col = layer.check_dimension(period_field, metric)
    segment_col = layer.check_dimension(dimension, metric)

    clauses, params = layer.filters_sql(filters, [metric])
    # Compare as text, so '2018-07-01' matches a DATE month and '2018-Q3' a quarter.
    clauses.insert(0, f"CAST({period_col} AS VARCHAR) IN (?, ?)")
    params = [str(base), str(compare)] + params
    where = "\n  AND ".join(clauses)

    segment_sql = f"""SELECT CAST({period_col} AS VARCHAR) AS period,
       COALESCE(CAST({segment_col} AS VARCHAR), 'unknown') AS segment,
       {m.numerator} AS num,
       {m.denominator} AS den
FROM {m.source}
WHERE {where}
GROUP BY ALL"""
    # Totals are queried separately, not summed from segments: an order with
    # items from two categories is in both category segments but only once here.
    total_sql = f"""SELECT CAST({period_col} AS VARCHAR) AS period,
       {m.numerator} AS num, {m.denominator} AS den
FROM {m.source}
WHERE {where}
GROUP BY ALL"""

    seg = con.execute(segment_sql, params).df()
    tot = con.execute(total_sql, params).df().set_index("period")
    for p in (str(base), str(compare)):
        if p not in tot.index or not tot.loc[p, "den"]:
            raise SemanticError(f"No data for {period_field} = {p!r} with these filters.")

    N0, N1 = tot.loc[str(base), "den"], tot.loc[str(compare), "den"]
    R0, R1 = tot.loc[str(base), "num"] / N0, tot.loc[str(compare), "num"] / N1

    # One row per segment, with base and compare side by side; missing = 0.
    wide = seg.pivot_table(index="segment", columns="period", values=["num", "den"],
                           aggfunc="sum", fill_value=0)
    k0 = wide["num"].get(str(base), 0.0) * 1.0
    k1 = wide["num"].get(str(compare), 0.0) * 1.0
    n0 = wide["den"].get(str(base), 0.0) * 1.0
    n1 = wide["den"].get(str(compare), 0.0) * 1.0

    w0, w1 = n0 / N0, n1 / N1
    r0 = (k0 / n0).where(n0 > 0)
    r1 = (k1 / n1).where(n1 > 0)
    # A segment that exists in only one period has no rate in the other; use
    # its other-period rate, so all of its contribution shows up as mix.
    r0_filled, r1_filled = r0.fillna(r1), r1.fillna(r0)

    table = pd.DataFrame({
        "segment": wide.index,
        "volume_base": n0.astype(int),
        "volume_compare": n1.astype(int),
        "rate_base": r0,
        "rate_compare": r1,
        "contribution": k1 / N1 - k0 / N0,
        "rate_effect": (w0 + w1) / 2 * (r1_filled - r0_filled),
        "mix_effect": (r0_filled + r1_filled) / 2 * (w1 - w0),
    }).reset_index(drop=True)
    change = R1 - R0
    table["share_of_change"] = table["contribution"] / change if change else float("nan")

    # Biggest contributors in the direction of the change come first.
    direction = 1 if change >= 0 else -1
    table = table.sort_values("contribution", ascending=(direction < 0)).reset_index(drop=True)

    return Drilldown(
        metric=metric, dimension=dimension, period_field=period_field,
        base=str(base), compare=str(compare), base_value=R0, compare_value=R1,
        table=table, residual=change - table["contribution"].sum(),
        sql=segment_sql, params=params, base_volume=N0, compare_volume=N1,
    )


def display_table(d: Drilldown, top: int = 5) -> pd.DataFrame:
    """Top contributors in human units, for the UI and the LLM explanation.

    The LLM is given percentages and percentage points that are already
    computed, so it never has to convert fractions itself (it got that wrong
    in testing). volume_share_pct sits next to share_of_change_pct: a segment
    with 75% of the orders explaining 75% of the change is "broad-based", not
    "concentrated"."""
    is_rate = d.metric != "avg_delay_days"   # avg_delay_days is in days, not a rate
    scale = 100 if is_rate else 1
    t = d.table.head(top)
    out = pd.DataFrame({
        "segment": t["segment"],
        "volume_base": t["volume_base"],
        "volume_compare": t["volume_compare"],
        "volume_share_pct": 100 * t["volume_compare"] / d.compare_volume,
        ("rate_base_pct" if is_rate else "value_base"): scale * t["rate_base"],
        ("rate_compare_pct" if is_rate else "value_compare"): scale * t["rate_compare"],
        ("contribution_pp" if is_rate else "contribution"): scale * t["contribution"],
        ("rate_effect_pp" if is_rate else "rate_effect"): scale * t["rate_effect"],
        ("mix_effect_pp" if is_rate else "mix_effect"): scale * t["mix_effect"],
        "share_of_change_pct": 100 * t["share_of_change"],
    })
    return out.round(2)


LEAD_TIME_METRICS = ["avg_promised_days", "avg_days_to_deliver"]


def lead_time_drivers(con, layer: SemanticLayer, period_field: str, base: str,
                      compare: str, filters=()) -> pd.DataFrame | None:
    """Promised vs actual delivery time in both periods. A late rate can rise
    for two very different reasons: deliveries got slower, or the promised
    date got tighter. Segment rankings cannot tell them apart; this can.

    Returns None when the filters use a dimension these order-level metrics
    do not have (e.g. seller or category)."""
    try:
        q = layer.compile(
            LEAD_TIME_METRICS, dimensions=[period_field],
            filters=list(filters) + [{"field": period_field, "op": "in",
                                      "value": [str(base), str(compare)]}],
        )
    except SemanticError:
        return None
    return con.execute(q.sql, q.params).df()


def explain(d: Drilldown, top: int = 5) -> str:
    """Human-readable summary (used by the CLI demo; the agent uses the LLM)."""
    lines = [f"{d.metric} by {d.dimension}: {d.base} {d.base_value:.4f} -> "
             f"{d.compare} {d.compare_value:.4f} (change {d.change * 100:+.2f} pp)"]
    for _, row in d.table.head(top).iterrows():
        lines.append(
            f"  {row.segment[:34]:<35} {row.contribution * 100:+6.2f} pp "
            f"({row.share_of_change:>5.0%} of change; rate {row.rate_effect * 100:+.2f}, "
            f"mix {row.mix_effect * 100:+.2f})  volume {row.volume_base} -> {row.volume_compare}")
    if abs(d.residual) > 0.0005:
        # Only happens for line-level dimensions (category, seller): an order
        # with items from two sellers counts in both sellers' segments.
        lines.append(f"  (overlap from orders in several segments: {d.residual * 100:+.2f} pp)")
    return "\n".join(lines)


if __name__ == "__main__":
    from ldi.sql_guard import connect_readonly

    con, layer = connect_readonly(), SemanticLayer()
    sp = [{"field": "state", "op": "=", "value": "SP"}]
    for dim in ["category", "seller_state", "seller"]:
        print(explain(decompose(con, layer, "late_delivery_rate", dim, "quarter",
                                "2018-Q2", "2018-Q3", sp)), "\n")
    print("Promised vs actual delivery days:")
    print(lead_time_drivers(con, layer, "quarter", "2018-Q2", "2018-Q3", sp).to_string(index=False))
