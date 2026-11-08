"""Module 7: Streamlit UI for the Late-Delivery Investigator.

    streamlit run app/streamlit_app.py

Shows the answer first, then (in expanders) how it was produced: the route,
the generated SQL, the result table, and the policy excerpts that were cited.
A chart is drawn only when the result has a shape that a chart helps with.
"""
import sys
from pathlib import Path

# `streamlit run` puts app/ on the import path, not the project root; add the
# root so `import ldi` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt  # noqa: E402  (ships with Streamlit)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ldi.agent import AgentResult  # noqa: E402
from ldi.router import Assistant, AssistantResult  # noqa: E402

EXAMPLES = [
    "Why did on-time delivery drop in São Paulo in Q3 2018?",
    "What was the on-time delivery rate by month in 2018?",
    "Did São Paulo breach the delivery SLA in Q3 2018, and what must we do?",
    "Who do we escalate to if a state misses the SLA two months in a row?",
    "Which carrier caused the most delays in RJ?",
]
TIME_COLUMNS = {"month", "quarter", "date"}
BLUE, RED = "#2a78d6", "#e34948"  # single series / diverging pair (lowered vs raised)

st.set_page_config(page_title="Late-Delivery Investigator", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Loading the database, the metric catalogue and the embedding model...")
def get_assistant() -> Assistant:
    # Built once per server process and shared by all sessions: the LLM client,
    # the read-only DuckDB connection, the semantic layer and the RAG index.
    return Assistant()


# ------------------------------------------------------------------ helpers
def column_config(df: pd.DataFrame, assistant: Assistant) -> dict:
    """Show each metric column in its own unit (%, R$, days), using metrics.yaml."""
    formats = {"percent": "percent", "currency": "R$ %.2f", "days": "%.2f", "count": "localized"}
    config = {}
    for col in df.columns:
        metric = assistant.agent.layer.metrics.get(col)
        if metric and metric.format in formats:
            config[col] = st.column_config.NumberColumn(format=formats[metric.format])
    return config


def simple_chart(df: pd.DataFrame) -> alt.Chart | None:
    """One label column + numeric columns -> line over time, or a ranked bar chart.
    Anything else (several label columns, a single row) gets no chart."""
    if df is None or len(df) < 2:
        return None
    numeric = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    if len(labels) != 1 or not numeric:
        return None
    x, y = labels[0], numeric[0]  # chart the first metric; the table shows the rest
    data = df[[x, y]].copy()
    tooltip = [alt.Tooltip(x), alt.Tooltip(y, format=",.4~f")]
    if x in TIME_COLUMNS:
        data[x] = data[x].astype(str)
        return (alt.Chart(data).mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=64, color=BLUE))
                .encode(x=alt.X(f"{x}:O", title=x), y=alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
                        tooltip=tooltip).properties(height=320))
    data = data.head(15)
    data[x] = data[x].astype(str).str.slice(0, 24)  # seller ids are 32 characters long
    return (alt.Chart(data).mark_bar(color=BLUE, cornerRadiusEnd=4, height={"band": 0.7})
            .encode(y=alt.Y(f"{x}:N", sort="-x", title=None), x=alt.X(f"{y}:Q", title=y), tooltip=tooltip)
            .properties(height=max(160, 28 * len(data))))


def contribution_chart(table: pd.DataFrame, metric: str) -> alt.Chart:
    """Horizontal bars of each segment's contribution, red = raised the metric."""
    col = "contribution_pp" if "contribution_pp" in table.columns else "contribution"
    data = table.copy()
    data["segment"] = data["segment"].astype(str).str.slice(0, 24)
    data["effect"] = data[col].map(lambda v: f"raised {metric}" if v >= 0 else f"lowered {metric}")
    return (alt.Chart(data).mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
            .encode(y=alt.Y("segment:N", sort=alt.EncodingSortField(col, op="sum", order="descending"), title=None),
                    x=alt.X(f"{col}:Q", title="contribution to the change (percentage points)"
                            if col.endswith("_pp") else "contribution to the change"),
                    color=alt.Color("effect:N", title=None, legend=alt.Legend(orient="top"),
                                    scale=alt.Scale(domain=[f"raised {metric}", f"lowered {metric}"],
                                                    range=[RED, BLUE])),
                    tooltip=[c for c in data.columns if c not in ("effect",)])
            .properties(height=max(160, 30 * len(data))))


# ---------------------------------------------------------------- sections
def show_data(r: AgentResult, assistant: Assistant) -> None:
    plan_type = r.plan.get("type", "?")
    st.markdown(f"**Data** · plan: `{plan_type}` · status: `{r.status}` · attempts: {r.attempts}")
    if r.plan.get("interpretation"):
        st.caption(f"Interpreted as: {r.plan['interpretation']}")

    if r.drilldowns:  # "why" question: one tab per dimension
        from ldi.drilldown import display_table
        tabs = st.tabs([f"by {d.dimension}" for d in r.drilldowns] + ["lead times"])
        for tab, d in zip(tabs, r.drilldowns):
            with tab:
                st.caption(f"{d.metric}: {d.base} {d.base_value:.2%} → {d.compare} {d.compare_value:.2%} "
                           f"({d.change * 100:+.2f} pp). Top 10 contributors:")
                top = display_table(d, top=10)
                st.altair_chart(contribution_chart(top, d.metric), width="stretch")
                st.dataframe(top, hide_index=True, width="stretch")
        with tabs[-1]:
            if r.lead_times is not None:
                st.caption("Promised vs actual delivery time (days). If promises shrink, "
                           "the late rate can rise even when deliveries get faster.")
                st.dataframe(r.lead_times.round(2), hide_index=True)
            else:
                st.caption("Not available with these filters.")
    elif r.data is not None:
        chart = simple_chart(r.data)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    if r.sql or r.errors:
        with st.expander("Generated SQL and result table"):
            for e in r.errors:
                st.warning(f"Attempt failed and the error was fed back to the LLM: {e}")
            if r.sql:
                st.code(r.sql, language="sql")
                if r.params:
                    st.caption(f"Parameters: {r.params}")
            if r.data is not None:
                st.dataframe(r.data, hide_index=True, width="stretch",
                             column_config=column_config(r.data, assistant))
                if r.truncated:
                    st.caption("Result truncated to the row limit.")


def show_policy(policy) -> None:
    st.markdown(f"**Policies** · status: `{policy.status}`")
    with st.expander(f"Policy sections given to the LLM ({len(policy.chunks)})"):
        if not policy.chunks:
            st.caption("No excerpt was close enough to the question, so the LLM was not called.")
        for c in policy.chunks:
            # Sections without a distance were added because their policy matched.
            how = f"matched, distance {c.distance:.3f}" if c.distance is not None else "same policy"
            st.markdown(f"**{c.citation}** · {how}")
            st.text(c.text)


def ask(question: str) -> None:
    st.session_state.question = question
    st.session_state.pending = True


# --------------------------------------------------------------------- page
st.title("📦 Late-Delivery Investigator")
st.caption("Ask about late deliveries in the Olist marketplace (orders Sep 2016 – Aug 2018) "
           "or about the operations policies. Answers come only from the data and the policy documents.")

try:
    assistant = get_assistant()
except Exception as e:  # e.g. missing .env key or Chroma index not built yet
    st.error(f"Could not start: {e}\n\nCheck .env, then run `python -m ldi.loader` "
             "and `python -m ldi.rag build`.")
    st.stop()

with st.sidebar:
    st.subheader("Setup")
    st.markdown(f"LLM: `{assistant.llm.provider}` / `{assistant.llm.model}`")
    st.markdown("Data: orders purchased 2016-09 to 2018-08 (later months are incomplete).")
    with st.expander("Data-quality checks"):
        dq = assistant.agent.con.execute(
            "SELECT status, check_name, bad_rows FROM dq_results ORDER BY status, check_name").df()
        st.dataframe(dq, hide_index=True)
    with st.expander("Metric catalogue"):
        st.text(assistant.agent.layer.describe())

st.markdown("**Try:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    col.button(example, on_click=ask, args=(example,), width="stretch")

with st.form("ask"):
    question = st.text_input("Your question", key="question",
                             placeholder="e.g. Which categories had the highest late delivery rate in 2018?")
    if st.form_submit_button("Ask", type="primary"):
        st.session_state.pending = True

if st.session_state.get("pending") and st.session_state.get("question"):
    st.session_state.pending = False
    with st.spinner("Routing, querying and explaining..."):
        try:
            st.session_state.result = assistant.ask(st.session_state.question)
        except Exception as e:  # rate limits, network errors: show, don't crash
            st.session_state.result = None
            st.error(f"Something went wrong: {type(e).__name__}: {e}")

result: AssistantResult | None = st.session_state.get("result")
if result:
    st.divider()
    st.markdown(f"### {result.question}")
    st.caption(f"Route: **{result.route}**")
    st.markdown(result.answer)
    if result.data:
        show_data(result.data, assistant)
    if result.policy:
        show_policy(result.policy)
