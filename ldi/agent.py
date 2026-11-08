"""Module 4: the text-to-SQL agent.

    question --LLM--> JSON plan --> SQL --> DuckDB (read-only) --> result --LLM--> answer

The plan is one of three kinds:
  metric         - use the semantic layer (preferred; definitions are fixed)
  sql            - a raw SELECT, only when no metric fits; checked by sql_guard
  cannot_answer  - the data does not contain what the question needs

If compiling or running the plan fails, the error message is sent back to the
LLM once so it can correct itself. Usage:

    python -m ldi.agent "What was the on-time delivery rate in SP in 2018-Q3?"
"""
import sys
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from ldi.llm import LLM, parse_json
from ldi.semantic import SemanticError, SemanticLayer
from ldi.sql_guard import UnsafeSQLError, connect_readonly, run_query, validate_sql

MAX_ATTEMPTS = 2  # first try + one retry with the error fed back
SCHEMA_TABLES = ["order_delivery", "orders_enriched", "reviews"]
CANNOT_ANSWER = "I can't answer that from this data"

PLAN_PROMPT = """You are the query planner of "Late-Delivery Investigator", an analytics
assistant for an operations manager at a Brazilian e-commerce marketplace (Olist).
Turn the user's question into ONE query plan, returned as a JSON object.

DATA COVERAGE
- Orders purchased from 2016-09-04 to 2018-10-17. After 2018-08-31 the data is
  incomplete (20 orders, none delivered), so 2018-Q3 effectively means July-August 2018.
- The data has NO information about: carrier or courier names, warehouses, routes,
  weather, costs or profit, marketing, stock levels, customer demographics, or
  anything after October 2018.

{semantic}

VIEWS AND TABLES (for raw SQL only)
{schema}

KNOWN VALUES
- quarters: {quarters}
- order_status: {statuses}
- customer states: {states}
- categories: {categories}

REPLY WITH EXACTLY ONE JSON OBJECT, one of:
1. Preferred, whenever a metric above fits the question:
   {{"interpretation": "<the question restated precisely>", "type": "metric",
     "metrics": ["<metric>", ...], "dimensions": ["<dimension>", ...],
     "filters": [{{"field": "<dimension>", "op": "=|!=|>|>=|<|<=|in|between", "value": ...}}],
     "min_orders": null, "limit": 100}}
2. Only if no metric fits, one DuckDB SELECT over the views/tables above:
   {{"interpretation": "...", "type": "sql", "sql": "SELECT ..."}}
3. If the data cannot answer the question:
   {{"interpretation": "...", "type": "cannot_answer", "reason": "<what is missing>"}}

RULES
- Never invent metrics, dimensions, columns, tables or values that are not listed above.
  If the question needs something that is not there, use cannot_answer. Do not guess.
- Filter values must match stored values exactly: states are 2-letter codes, quarters
  look like '2018-Q3', months are dates like '2018-07-01'.
- "São Paulo" means the state SP unless the question clearly says the city.
- Metrics from different views cannot be combined in one plan; pick the metrics the
  question needs most.
- When ranking sellers or categories by a rate, set min_orders (e.g. 30) so tiny
  groups do not dominate.
- In raw SQL on orders_enriched, count orders with count(DISTINCT order_id), never count(*).
"""

EXPLAIN_PROMPT = """You explain query results to an operations manager in plain English.
- Use ONLY the numbers in the result table. Do not invent causes, numbers or context
  that the table does not show.
- Rates are fractions: 0.934 means 93.4%. Round sensibly.
- 2-4 sentences, direct answer first.
- If the table is empty, say that no matching data was found.
- If the result was truncated, say that only the first rows are shown."""


@dataclass
class AgentResult:
    question: str
    status: str = "error"           # ok | cannot_answer | error
    answer: str = ""
    plan: dict = field(default_factory=dict)
    sql: str = ""
    params: list = field(default_factory=list)
    data: pd.DataFrame | None = None
    truncated: bool = False
    attempts: int = 0
    errors: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLM | None = None, con: duckdb.DuckDBPyConnection | None = None,
                 layer: SemanticLayer | None = None):
        self.llm = llm or LLM()
        self.con = con or connect_readonly()
        self.layer = layer or SemanticLayer()
        self.system_prompt = self._build_prompt()

    # ------------------------------------------------------------- prompt
    def _build_prompt(self) -> str:
        def values(sql: str) -> str:
            return ", ".join(str(r[0]) for r in self.con.execute(sql).fetchall())

        schema = []
        for table in SCHEMA_TABLES:
            cols = self.con.execute(f"DESCRIBE {table}").fetchall()
            schema.append(f"{table}(" + ", ".join(f"{c[0]} {c[1]}" for c in cols) + ")")

        return PLAN_PROMPT.format(
            semantic=self.layer.describe(),
            schema="\n".join(schema),
            quarters=values("SELECT DISTINCT purchase_quarter FROM order_delivery ORDER BY 1"),
            statuses=values("SELECT DISTINCT order_status FROM order_delivery ORDER BY 1"),
            states=values("SELECT DISTINCT customer_state FROM order_delivery ORDER BY 1"),
            categories=values("SELECT DISTINCT category FROM orders_enriched ORDER BY 1"),
        )

    # ---------------------------------------------------------------- ask
    def ask(self, question: str) -> AgentResult:
        result = AgentResult(question=question)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts = attempt
            reply = self.llm.chat(messages, json_mode=True)
            try:
                plan = parse_json(reply)
                result.plan = plan
                if plan.get("type") == "cannot_answer":
                    result.status = "cannot_answer"
                    result.answer = f"{CANNOT_ANSWER}: {plan.get('reason', 'the data does not cover it')}"
                    return result
                result.sql, result.params = self._plan_to_sql(plan)
                result.data, result.truncated = run_query(self.con, result.sql, result.params)
                break  # success
            except (ValueError, duckdb.Error) as e:
                # ValueError covers bad JSON, SemanticError and UnsafeSQLError.
                error = f"{type(e).__name__}: {e}"
                result.errors.append(error)
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"That plan failed with this error:\n{error}\n"
                                                "Return a corrected JSON plan."},
                ]
        else:  # no break: every attempt failed
            result.answer = (f"{CANNOT_ANSWER}: the query failed after {MAX_ATTEMPTS} attempts "
                             f"({result.errors[-1]}).")
            return result

        result.status = "ok"
        result.answer = self._explain(result)
        return result

    def _plan_to_sql(self, plan: dict) -> tuple[str, list]:
        kind = plan.get("type")
        if kind == "metric":
            q = self.layer.compile(
                plan.get("metrics", []),
                dimensions=plan.get("dimensions") or [],
                filters=plan.get("filters") or [],
                min_orders=plan.get("min_orders"),
                limit=plan.get("limit") or 100,
            )
            return q.sql, q.params
        if kind == "sql":
            return validate_sql(plan.get("sql", "")), []
        raise ValueError(f"Plan 'type' must be metric, sql or cannot_answer, got {kind!r}.")

    def _explain(self, result: AgentResult) -> str:
        table = result.data.head(50).to_csv(index=False)
        note = " (truncated: only the first rows were returned)" if result.truncated else ""
        content = (f"Question: {result.question}\n"
                   f"Interpretation: {result.plan.get('interpretation', '')}\n"
                   f"SQL:\n{result.sql}\nParameters: {result.params}\n"
                   f"Result ({len(result.data)} rows{note}):\n{table}")
        return self.llm.chat([
            {"role": "system", "content": EXPLAIN_PROMPT},
            {"role": "user", "content": content},
        ])


def print_result(r: AgentResult) -> None:
    # The Windows console defaults to cp1252, which cannot print characters
    # LLMs like to use (narrow spaces, "≈", accents). Switch stdout to UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\nQ: {r.question}")
    print(f"Status: {r.status} | attempts: {r.attempts} | plan type: {r.plan.get('type')}")
    if r.plan.get("interpretation"):
        print(f"Interpretation: {r.plan['interpretation']}")
    for e in r.errors:
        print(f"  error fed back: {e}")
    if r.sql:
        print(f"SQL:\n{r.sql}\nParams: {r.params}")
    if r.data is not None:
        print(r.data.head(10).to_string(index=False))
    print(f"\nA: {r.answer}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m ldi.agent "your question"')
    print_result(Agent().ask(" ".join(sys.argv[1:])))
