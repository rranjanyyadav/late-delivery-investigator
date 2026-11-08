"""Module 8: evaluate the text-to-SQL agent against eval/questions.yaml.

    python eval/run_eval.py                 # all questions
    python eval/run_eval.py --label baseline
    python eval/run_eval.py --only q11_review_late_vs_on_time q14_freight_per_order_sp_rj

Grading is EXECUTION-based: we compare the agent's result table with the gold
SQL's result, not the SQL text (two different queries can both be right).

Rules for a gold_sql question:
  - same number of rows;
  - every gold column must match some column of the agent's result (extra
    columns, such as a state column next to the rate, are fine);
  - columns are compared as sorted lists (row order and column names do not
    matter); numbers with a relative tolerance of 0.1%, and a rate given as a
    percentage (93.2) matches the same rate as a fraction (0.932); dates are
    compared as YYYY-MM-DD text.

Every failure gets ONE error category, decided by fixed rules (see categorise),
so the breakdown is reproducible rather than a judgement call.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import ldi` works when run as a script

import openai  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ldi.agent import Agent, AgentResult  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.yaml"
RESULTS_DIR = ROOT / "eval" / "results"
CATEGORIES = ["wrong metric", "wrong join/filter", "misread question",
              "hallucinated column", "execution error"]
API_ERROR = "api error"  # not an agent error; counted separately
HALLUCINATION_MARKERS = ["Binder Error", "not found", "does not exist",
                         "Unknown metric", "Unknown dimension", "Referenced column"]


# ------------------------------------------------------------ comparing
def _normalise(series: pd.Series):
    """Return ('num', sorted floats) or ('text', sorted strings)."""
    s = series.dropna()
    if pd.api.types.is_bool_dtype(s):
        return "text", sorted(str(v).lower() for v in s)
    if pd.api.types.is_numeric_dtype(s):
        return "num", sorted(float(v) for v in s)
    return "text", sorted(str(v)[:10] if pd.api.types.is_datetime64_any_dtype(series)
                          else str(v).strip().lower() for v in s)


def _numbers_match(gold: list[float], pred: list[float]) -> bool:
    if len(gold) != len(pred):
        return False
    for scale in (1, 100, 0.01):  # fraction vs percentage
        scaled = sorted(p / scale for p in pred)
        if all(math.isclose(g, p, rel_tol=1e-3, abs_tol=1e-6) for g, p in zip(gold, scaled)):
            return True
    return False


def columns_match(gold_col: pd.Series, pred_col: pd.Series) -> bool:
    g_kind, g = _normalise(gold_col)
    p_kind, p = _normalise(pred_col)
    if g_kind != p_kind:  # dates are already text on both sides, so this is a real mismatch
        return False
    return _numbers_match(g, p) if g_kind == "num" else g == p


def results_match(gold: pd.DataFrame, pred: pd.DataFrame) -> bool:
    if pred is None or len(gold) != len(pred):
        return False
    unused = list(pred.columns)
    for gcol in gold.columns:
        match = next((c for c in unused if columns_match(gold[gcol], pred[c])), None)
        if match is None:
            return False
        unused.remove(match)  # each agent column can satisfy only one gold column
    return True


def is_empty(df: pd.DataFrame | None) -> bool:
    return df is None or len(df) == 0 or df.isna().all().all()


# -------------------------------------------------------------- grading
def grade(q: dict, r: AgentResult, con) -> tuple[bool, str | None]:
    """Return (correct, error_category or None)."""
    expect = q.get("expect")
    if expect == "cannot_answer":
        return (True, None) if r.status == "cannot_answer" else (False, "misread question")
    if expect == "no_data":
        ok = r.status == "cannot_answer" or (r.status == "ok" and is_empty(r.data))
        return (True, None) if ok else (False, "misread question")

    if r.status == "cannot_answer":
        return False, "misread question"   # refused an answerable question
    if r.status != "ok":
        last = r.errors[-1] if r.errors else ""
        if any(m.lower() in last.lower() for m in HALLUCINATION_MARKERS):
            return False, "hallucinated column"
        return False, "execution error"

    if "drilldown" in q:
        want = q["drilldown"]
        if r.plan.get("type") != "drilldown":
            return False, "misread question"
        for d in r.drilldowns:
            if d.dimension == want["dimension"]:
                top = str(d.table.iloc[0]["segment"])
                return (True, None) if top == want["top_segment"] else (False, "wrong join/filter")
        return False, "misread question"   # never drilled into the expected dimension

    gold = con.execute(q["gold_sql"]).df()
    if results_match(gold, r.data):
        return True, None
    if q.get("metric") and r.plan.get("type") == "metric" \
            and q["metric"] not in (r.plan.get("metrics") or []):
        return False, "wrong metric"
    if r.data is None or len(r.data) != len(gold):
        return False, "misread question"   # wrong grouping, period or top-N
    return False, "wrong join/filter"      # right shape, wrong numbers


# ------------------------------------------------------------------ run
def run(label: str, only: list[str] | None = None) -> dict:
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if only:
        questions = [q for q in questions if q["id"] in only]
    agent = Agent()
    rows = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        try:
            r = agent.ask(q["question"], explain=False)
            correct, category = grade(q, r, agent.con)
        except openai.APIError as e:
            # Rate limit / network: the model never saw the question, so this is
            # not an agent error. Reported separately, never silently dropped.
            r, correct, category = AgentResult(q["question"], errors=[repr(e)]), False, API_ERROR
        except Exception as e:  # any other crash is the agent's fault
            r, correct, category = AgentResult(q["question"], errors=[repr(e)]), False, "execution error"
        rows.append({
            "id": q["id"], "correct": correct, "category": category,
            "plan_type": r.plan.get("type"), "metrics": r.plan.get("metrics") or r.plan.get("metric"),
            "attempts": r.attempts, "recovered_by_retry": correct and r.attempts > 1,
            "interpretation": r.plan.get("interpretation"), "sql": r.sql, "params": r.params,
            "errors": r.errors, "rows": None if r.data is None else len(r.data),
            "seconds": round(time.time() - start, 1), "tags": q.get("tags", []),
        })
        mark = "PASS" if correct else f"FAIL ({category})"
        print(f"[{i:>2}/{len(questions)}] {q['id']:<36} {mark}", flush=True)

    n_correct = sum(r["correct"] for r in rows)
    n_api = sum(r["category"] == API_ERROR for r in rows)
    graded = len(rows) - n_api
    report = {
        "label": label, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": f"{agent.llm.provider}/{agent.llm.model}",
        "total": len(rows), "correct": n_correct,
        "accuracy": round(n_correct / len(rows), 4) if rows else 0,
        "api_errors": n_api,
        "accuracy_excluding_api_errors": round(n_correct / graded, 4) if graded else 0,
        "errors": {c: sum(r["category"] == c for r in rows) for c in CATEGORIES},
        "recovered_by_retry": sum(r["recovered_by_retry"] for r in rows),
        "questions": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{label}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n{label}: {n_correct}/{len(rows)} correct = {report['accuracy']:.0%}  ({report['model']})")
    if n_api:
        print(f"API errors (rate limit/network, not graded): {n_api} -> "
              f"{n_correct}/{graded} = {report['accuracy_excluding_api_errors']:.0%} of graded questions")
    print("Errors: " + ", ".join(f"{c}: {n}" for c, n in report["errors"].items()))
    print(f"Answered correctly only thanks to the retry: {report['recovered_by_retry']}")
    print(f"Saved {path.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--only", nargs="*", help="question ids to run")
    args = parser.parse_args()
    run(args.label, args.only)
