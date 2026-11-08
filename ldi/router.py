"""Module 6: route a question to the SQL agent, the policy RAG, or both.

    data    -> Agent (modules 4-5): numbers, trends, "why did X change"
    policy  -> RAG: SLAs, penalties, escalation rules
    both    -> split the question into a data part and a policy part, answer
               each, then combine ("did SP breach the SLA, and what happens next?")

    python -m ldi.router "Did SP breach the on-time SLA in Q3 2018 and who must act?"
"""
import sys
from dataclasses import dataclass

from ldi import rag
from ldi.agent import Agent, AgentResult, print_result
from ldi.llm import LLM, parse_json

ROUTER_PROMPT = """You route questions for an e-commerce operations assistant.
Routes:
- "data":   needs numbers from the orders database: delivery rates, counts, delays,
            revenue, trends, rankings, or WHY a metric changed.
- "policy": about company rules in the policy documents listed below: targets, SLAs,
            penalties, escalation, procedures. No numbers from the database needed.
- "both":   needs data AND a policy, e.g. "did SP breach the SLA in Q3 2018?" (needs
            the actual rate and the SLA target) or "what should we do about X?".
Policy documents: {titles}

For "both": the database has NO targets, SLAs or policies, so the data_question must
ask only for numbers (never mention SLAs, targets, breaches or policies in it). When
checking a target, ask for the metric in the period AND the previous period, because
policies can have period-over-period rules. Example:
  "Did SP breach the delivery SLA in Q3 2018?" ->
  data_question: "What was the on-time delivery rate in SP in 2018-Q2 and 2018-Q3?"
  policy_question: "What are the on-time delivery targets and when is an SLA breach recorded?"

Return JSON: {{"route": "data|policy|both",
  "data_question": "<self-contained question for the database, or empty>",
  "policy_question": "<self-contained question for the policies, or empty>"}}"""

COMBINE_PROMPT = """Write one answer for an operations manager from two inputs: an answer
from the order data and an answer from the policy documents.
- State the data facts first, then which policy rules apply to them, and say plainly
  whether a target or threshold is met or breached and what action the policy requires.
- Compare numbers carefully (e.g. 93.4% vs a 90% target is MET).
- Keep the policy citations like [POL-01 §Targets]. Use only the two inputs.
- If either input says it cannot answer, say what is missing.
- 3-6 sentences."""


@dataclass
class AssistantResult:
    question: str
    route: str
    answer: str
    data: AgentResult | None = None
    policy: rag.RagResult | None = None


class Assistant:
    def __init__(self, llm: LLM | None = None, agent: Agent | None = None):
        self.llm = llm or LLM()
        self.agent = agent or Agent(llm=self.llm)
        self.router_prompt = ROUTER_PROMPT.format(titles="; ".join(rag.policy_titles()))

    def route(self, question: str) -> dict:
        try:
            plan = parse_json(self.llm.chat([
                {"role": "system", "content": self.router_prompt},
                {"role": "user", "content": question},
            ], json_mode=True))
        except ValueError:
            plan = {}
        if plan.get("route") not in ("data", "policy", "both"):
            plan["route"] = "data"  # safe default: the data path can still refuse
        # Fall back to the full question if the router left a part empty.
        plan["data_question"] = plan.get("data_question") or question
        plan["policy_question"] = plan.get("policy_question") or question
        return plan

    def ask(self, question: str) -> AssistantResult:
        plan = self.route(question)
        route = plan["route"]
        if route == "data":
            data = self.agent.ask(plan["data_question"])
            return AssistantResult(question, route, data.answer, data=data)
        if route == "policy":
            policy = rag.answer(plan["policy_question"], self.llm)
            return AssistantResult(question, route, policy.answer, policy=policy)

        data = self.agent.ask(plan["data_question"])
        policy = rag.answer(plan["policy_question"], self.llm)
        combined = self.llm.chat([
            {"role": "system", "content": COMBINE_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\n"
                                        f"From the data ({plan['data_question']}):\n{data.answer}\n\n"
                                        f"From the policies ({plan['policy_question']}):\n{policy.answer}"},
        ])
        return AssistantResult(question, route, combined, data=data, policy=policy)


def print_assistant(r: AssistantResult) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\nQ: {r.question}\nRoute: {r.route}")
    if r.data:
        print(f"  data part:   {r.data.question}  -> {r.data.status}, plan={r.data.plan.get('type')}")
    if r.policy:
        print(f"  policy part: {r.policy.question}  -> {r.policy.status}, "
              f"chunks={[c.citation for c in r.policy.chunks]}")
    print(f"\nA: {r.answer}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m ldi.router "your question"')
    print_assistant(Assistant().ask(" ".join(sys.argv[1:])))
