"""Module 3: the semantic layer. Turns "metric + dimensions + filters" into SQL.

    layer = SemanticLayer()
    q = layer.compile(["on_time_delivery_rate"], dimensions=["quarter"],
                      filters=[{"field": "state", "op": "=", "value": "SP"}])
    q.sql     -> SELECT purchase_quarter AS quarter, ... FROM orders_enriched WHERE customer_state = ? ...
    q.params  -> ['SP']

Why this exists: an LLM writing raw SQL can define "on-time rate" a slightly
different way every time. Here the definitions live in semantic/metrics.yaml,
and the LLM only chooses WHICH metric, dimensions and filters it needs.

Run `python -m ldi.semantic` for a demo against the real database.
"""
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

from ldi.config import DB_PATH, METRICS_PATH

OPERATORS = {"=", "!=", ">", ">=", "<", "<=", "in", "between"}
TIME_DIMENSIONS = {"date", "month", "quarter"}
MAX_LIMIT = 1000


class SemanticError(ValueError):
    """A request the semantic layer cannot compile. The message is written so
    that it can be fed back to the LLM, which can then fix its request."""


@dataclass
class Dimension:
    name: str
    column: str
    description: str


@dataclass
class Metric:
    name: str
    description: str
    source: str
    grain: str
    expression: str
    format: str
    dimensions: list[str]


@dataclass
class Query:
    sql: str
    params: list  # values for the ? placeholders, in order


class SemanticLayer:
    def __init__(self, path: Path = METRICS_PATH):
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.dimensions = {
            name: Dimension(name=name, **d) for name, d in spec["dimensions"].items()
        }
        self.metrics = {
            name: Metric(name=name, **m) for name, m in spec["metrics"].items()
        }
        # Catch typos in the YAML at start-up, not in the middle of a query.
        for m in self.metrics.values():
            unknown = set(m.dimensions) - set(self.dimensions)
            if unknown:
                raise ValueError(f"metrics.yaml: {m.name} uses unknown dimensions {unknown}")

    # ------------------------------------------------------------------ compile
    def compile(self, metrics, dimensions=(), filters=(), min_orders=None,
                limit=100) -> Query:
        """Build one SELECT for one or more metrics that share a source view.

        metrics:    metric name or list of names
        dimensions: names to GROUP BY, e.g. ["state", "month"]
        filters:    list of {"field": <dimension>, "op": <operator>, "value": ...}
        min_orders: drop groups with fewer orders (rates on tiny groups are noise)
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        if not metrics:
            raise SemanticError("Ask for at least one metric.")
        chosen = [self._get_metric(name) for name in metrics]

        sources = {m.source for m in chosen}
        if len(sources) > 1:
            raise SemanticError(
                f"Metrics {list(metrics)} come from different views {sorted(sources)}; "
                "query them separately."
            )
        source = sources.pop()

        # A dimension is usable only if EVERY requested metric allows it.
        allowed = set.intersection(*(set(m.dimensions) for m in chosen))
        for d in dimensions:
            self._check_dimension(d, allowed, metrics)

        select = [f"{self.dimensions[d].column} AS {d}" for d in dimensions]
        select += [f"{' '.join(m.expression.split())} AS {m.name}" for m in chosen]

        where, params = [], []
        for f in filters:
            clause, values = self._compile_filter(f, allowed, metrics)
            where.append(clause)
            params.extend(values)

        sql = "SELECT " + ",\n       ".join(select) + f"\nFROM {source}"
        if where:
            sql += "\nWHERE " + "\n  AND ".join(where)
        # Group/order by the real column names, not the aliases: aliases such
        # as "date" or "month" are also SQL keywords and type names.
        columns = {d: self.dimensions[d].column for d in dimensions}
        if dimensions:
            sql += "\nGROUP BY " + ", ".join(columns.values())
        if min_orders:
            sql += f"\nHAVING count(DISTINCT order_id) >= {int(min_orders)}"

        # Time series read left to right in time order; everything else is a
        # ranking, so the biggest value comes first.
        time_dims = [d for d in dimensions if d in TIME_DIMENSIONS]
        if time_dims:
            sql += "\nORDER BY " + ", ".join(columns[d] for d in time_dims)
        elif dimensions:
            sql += f"\nORDER BY {chosen[0].name} DESC NULLS LAST"

        sql += f"\nLIMIT {max(1, min(int(limit), MAX_LIMIT))}"
        return Query(sql, params)

    def _get_metric(self, name: str) -> Metric:
        if name not in self.metrics:
            raise SemanticError(
                f"Unknown metric '{name}'. Available metrics: {', '.join(self.metrics)}."
            )
        return self.metrics[name]

    def _check_dimension(self, name: str, allowed: set, metrics) -> None:
        if name not in self.dimensions:
            raise SemanticError(
                f"Unknown dimension '{name}'. Available dimensions: {', '.join(self.dimensions)}."
            )
        if name not in allowed:
            raise SemanticError(
                f"Dimension '{name}' is not allowed for {list(metrics)}. "
                f"Allowed: {', '.join(sorted(allowed))}."
            )

    def _compile_filter(self, f: dict, allowed: set, metrics) -> tuple[str, list]:
        field, op, value = f.get("field"), str(f.get("op", "=")).lower(), f.get("value")
        self._check_dimension(field, allowed, metrics)
        if op not in OPERATORS:
            raise SemanticError(f"Unknown operator '{op}'. Use one of: {', '.join(sorted(OPERATORS))}.")
        column = self.dimensions[field].column

        # Values are ALWAYS passed as ? parameters, never pasted into the SQL
        # string, so a value like "SP' OR 1=1 --" cannot change the query.
        if op == "in":
            if not isinstance(value, list) or not value:
                raise SemanticError(f"Filter on '{field}' with 'in' needs a non-empty list.")
            return f"{column} IN ({', '.join('?' * len(value))})", list(value)
        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise SemanticError(f"Filter on '{field}' with 'between' needs [low, high].")
            return f"{column} BETWEEN ? AND ?", list(value)
        if isinstance(value, list):
            raise SemanticError(f"Filter on '{field}' with '{op}' needs a single value.")
        return f"{column} {op} ?", [value]

    # ------------------------------------------------------------- for the LLM
    def describe(self) -> str:
        """Plain-text catalogue of dimensions and metrics, for the LLM prompt."""
        lines = ["DIMENSIONS (group by / filter on these names):"]
        for d in self.dimensions.values():
            lines.append(f"- {d.name}: {d.description}")
        lines.append("\nMETRICS:")
        for m in self.metrics.values():
            desc = " ".join(m.description.split())
            lines.append(
                f"- {m.name} [{m.format}, view={m.source}, grain={m.grain}]: {desc} "
                f"Dimensions: {', '.join(m.dimensions)}."
            )
        return "\n".join(lines)


def run(con: duckdb.DuckDBPyConnection, query: Query):
    """Execute a compiled query and return a pandas DataFrame."""
    return con.execute(query.sql, query.params).df()


if __name__ == "__main__":
    layer = SemanticLayer()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("Every metric, whole dataset:")
    for name in layer.metrics:
        value = run(con, layer.compile(name)).iloc[0, 0]
        print(f"  {name:<28}{value:>18,.4f}")

    examples = [
        ("On-time rate in SP, 2018-Q2 vs 2018-Q3",
         dict(metrics=["on_time_delivery_rate", "delivered_orders"], dimensions=["quarter"],
              filters=[{"field": "state", "op": "=", "value": "SP"},
                       {"field": "quarter", "op": "in", "value": ["2018-Q2", "2018-Q3"]}])),
        ("Top 5 categories by late-delivery rate in SP, 2018-Q3 (>= 100 orders)",
         dict(metrics=["late_delivery_rate", "late_order_count", "delivered_orders"],
              dimensions=["category"], min_orders=100, limit=5,
              filters=[{"field": "state", "op": "=", "value": "SP"},
                       {"field": "quarter", "op": "=", "value": "2018-Q3"}])),
    ]
    for title, request in examples:
        q = layer.compile(**request)
        print(f"\n{title}\n{q.sql}\n  params={q.params}")
        print(run(con, q).to_string(index=False))

    print("\nA request the layer must refuse:")
    try:
        layer.compile("avg_delay_days", dimensions=["seller"])
    except SemanticError as e:
        print("  SemanticError:", e)
    con.close()
