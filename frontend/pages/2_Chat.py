"""Chat Interface — natural language interaction with all data engineering agents."""

import streamlit as st
import pandas as pd
import plotly.express as px
from api import get, post

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")
st.title("💬 Chat with Your Data")
st.caption("Ask questions in plain English — the orchestrator routes to the right agent.")

# ── Sidebar ───────────────────────────────────────────────
health = get("/health") or {}
agent_list = health.get("agents", [])

with st.sidebar:
    st.subheader("🤖 Available Agents")
    for a in agent_list:
        st.markdown(f"- **{a['name'].title()}**: {a['description'][:60]}...")

    st.divider()
    st.subheader("💡 Example Questions")
    examples = [
        "Show me top 5 products by revenue",
        "Monthly revenue trend",
        "Run full ETL lifecycle (profile, clean, build marts)",
        "Scan for anomalies and PII",
        "Generate an executive summary",
        "What's our repeat customer rate?",
        "Describe the data schema",
        "Generate a data dictionary",
        "Revenue by country and channel segmentation",
        "Run everything — all agents",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, type="secondary"):
            st.session_state["chat_input"] = ex

# ── Chat history ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm the DataEng Copilot. Ask me anything about your data — SQL queries, ETL pipelines, quality scans, reports, or documentation."}
    ]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "artifacts" in msg:
            for art in msg["artifacts"]:
                if art.get("type") == "chart":
                    df = pd.DataFrame({
                        art.get("x_label", "x"): art["x"],
                        art.get("y_label", "y"): art["y"],
                    })
                    fn = px.line if art["chart_type"] == "line" else px.bar
                    st.plotly_chart(fn(df, x=df.columns[0], y=df.columns[1],
                                       title=art["title"]), use_container_width=True)

# ── Input ─────────────────────────────────────────────────
prompt = st.session_state.get("chat_input", "")
if prompt:
    del st.session_state["chat_input"]
else:
    prompt = st.chat_input("Ask a data question...")

if prompt:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call orchestrator
    resp = post("/agents/run", {"agent": "", "input": prompt})
    with st.chat_message("assistant"):
        if resp and resp.get("status") != "error":
            detail = resp.get("detail", "")
            summary = resp.get("summary", "")
            sql = resp.get("sql")
            artifacts = resp.get("artifacts", [])

            # Show summary
            st.markdown(f"**{summary}**")

            # Show SQL if present
            if sql:
                with st.expander("🔍 SQL Query", expanded=False):
                    st.code(sql, language="sql")

            # Show detail in expander
            if detail and len(detail) > 200:
                with st.expander("📄 Full Results", expanded=False):
                    st.markdown(detail)
            elif detail:
                st.markdown(detail)

            # Show artifacts (charts)
            for art in artifacts:
                if art.get("type") == "chart":
                    df = pd.DataFrame({
                        art.get("x_label", "x"): art["x"],
                        art.get("y_label", "y"): art["y"],
                    })
                    fn = px.line if art["chart_type"] == "line" else px.bar
                    st.plotly_chart(fn(df, x=df.columns[0], y=df.columns[1],
                                       title=art["title"]), use_container_width=True)

            # Save for display
            msg_data = {"role": "assistant", "content": summary}
            if artifacts:
                msg_data["artifacts"] = artifacts
            st.session_state["messages"].append(msg_data)
        else:
            err = resp.get("error", "Unknown error") if resp else "Backend unavailable"
            st.error(f"Error: {err}")
            st.session_state["messages"].append({"role": "assistant", "content": f"❌ {err}"})

    # Clear the temp prompt
    st.session_state["chat_input"] = ""