"""Agent control flow with a scripted fake LLM: no network, no API key."""
from conftest import FakeLLM

from ldi.agent import CANNOT_ANSWER, Agent


def make_agent(con, *replies):
    llm = FakeLLM(*replies)
    return Agent(llm=llm, con=con), llm


def test_metric_plan_runs_through_the_semantic_layer(con):
    agent, _ = make_agent(con, '{"type": "metric", "metrics": ["delivered_orders"]}', "explained")
    r = agent.ask("How many orders were delivered?")
    assert r.status == "ok" and r.data.iloc[0, 0] == 3 and r.answer == "explained"


def test_error_is_fed_back_and_the_retry_succeeds(con):
    agent, llm = make_agent(
        con,
        '{"type": "metric", "metrics": ["avg_delay_days"], "dimensions": ["seller"]}',  # not allowed
        '{"type": "metric", "metrics": ["avg_delay_days"], "dimensions": ["state"]}',
        "explained",
    )
    r = agent.ask("Average delay by seller?")
    assert r.status == "ok" and r.attempts == 2
    assert "not allowed" in llm.calls[1][-1]["content"]   # the error text reached the LLM


def test_unsafe_sql_is_never_executed(con):
    drop = '{"type": "sql", "sql": "DROP TABLE orders"}'
    agent, _ = make_agent(con, drop, drop)
    r = agent.ask("Delete everything")
    assert r.status == "error" and r.answer.startswith(CANNOT_ANSWER)
    assert con.execute("SELECT count(*) FROM orders").fetchall()[0][0] == 4


def test_cannot_answer_is_passed_through(con):
    agent, llm = make_agent(con, '{"type": "cannot_answer", "reason": "no carrier data"}')
    r = agent.ask("Which carrier was slowest?")
    assert r.status == "cannot_answer"
    assert r.answer == f"{CANNOT_ANSWER}: no carrier data"
    assert len(llm.calls) == 1   # no explanation call for a refusal


def test_explain_false_skips_the_second_call(con):
    agent, llm = make_agent(con, '{"type": "metric", "metrics": ["revenue"]}')
    r = agent.ask("Revenue?", explain=False)
    assert r.status == "ok" and r.answer == "" and len(llm.calls) == 1
