"""The eval grader and the policy chunker (neither needs the LLM)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_eval import results_match  # noqa: E402

from ldi.config import POLICY_DIR  # noqa: E402
from ldi.rag import parse_policy  # noqa: E402

GOLD = pd.DataFrame({"state": ["SP", "RJ"], "rate": [0.25, 0.5]})


def test_grader_ignores_order_names_and_extra_columns():
    pred = pd.DataFrame({"n": [10, 20], "customer_state": ["RJ", "SP"], "late": [0.5, 0.25]})
    assert results_match(GOLD, pred)


def test_grader_accepts_percent_for_fraction():
    assert results_match(GOLD, pd.DataFrame({"s": ["SP", "RJ"], "pct": [25.0, 50.0]}))


def test_grader_rejects_wrong_values_and_wrong_row_count():
    assert not results_match(GOLD, pd.DataFrame({"s": ["SP", "RJ"], "r": [0.25, 0.51]}))
    assert not results_match(GOLD, pd.DataFrame({"s": ["SP"], "r": [0.25]}))


def test_policies_chunk_by_section_with_citations():
    chunks = parse_policy(POLICY_DIR / "POL-01_delivery_sla.md")
    sections = [c.section for c in chunks]
    assert "Targets" in sections and "SLA breach" in sections
    targets = next(c for c in chunks if c.section == "Targets")
    assert targets.citation == "[POL-01 §Targets]"
    assert targets.text.startswith("POL-01 Delivery SLA")   # title prefix for retrieval
    assert "90%" in targets.text
