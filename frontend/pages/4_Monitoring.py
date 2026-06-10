"""Agent Monitoring Dashboard — real-time view of agent activity, system health, and metrics."""

import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from api import get, post

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")
st.title("📊 Agent Monitoring Dashboard")
st.caption("Real-time metrics for all data engineering agents. Auto-refreshes every 5s.")

# ── Auto-refresh ──────────────────────────────────────────
if "monitor_running" not in st.session_state:
    st.session_state["monitor_running"] = False
if "metric_history" not in st.session_state:
    st.session_state["metric_history"] = pd.DataFrame(columns=[
        "timestamp", "agent", "status", "tokens_in", "tokens_out", "cost", "duration_s"
    ])
if "run_counter" not in st.session_state:
    st.session_state["run_counter"] = 0

# ── Health check ──────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ System Health")
    health = get("/health") or {}
    live_ok = health.get("live_available", False)
    rec_ok = health.get("recording_available", False)
    agent_count = health.get("agent_count", 0)

    st.metric("Backend Status", "✅ Online" if health else "❌ Offline")
    st.metric("Agents Registered", agent_count)
    st.metric("Live API Key", "✅ Set" if live_ok else "❌ Not set")
    st.metric("Recording", "✅ Available" if rec_ok else "❌ Not found")

    st.divider()
    st.subheader("🤖 Agents")
    for a in health.get("agents", []):
        st.markdown(f"- **{a['name'].title()}**")

    st.divider()
    st.subheader("▶️ Quick Run")

    quick_agents = {
        "Run SQL: Top Products": ("sql", "Show me top 5 products by revenue"),
        "Run ETL: Full Lifecycle": ("etl", "Run full ETL lifecycle"),
        "Quality Scan": ("quality", "Full quality scan including anomalies and PII"),
        "Analytics: Executive Summary": ("analytics", "Executive summary with KPIs"),
        "Data Dictionary": ("docs", "Generate data dictionary"),
        "🤖 Run All Agents": ("orchestrator", "Run everything - all agents"),
    }

    for label, (agent, inp) in quick_agents.items():
        if st.button(label, use_container_width=True, type="secondary"):
            resp = post("/agents/run", {"agent": agent, "input": inp})
            if resp:
                st.session_state["run_counter"] += 1
                new_row = pd.DataFrame([{
                    "timestamp": time.time(),
                    "agent": resp.get("agent", agent),
                    "status": resp.get("status", "?"),
                    "tokens_in": resp.get("tokens_in", 0),
                    "tokens_out": resp.get("tokens_out", 0),
                    "cost": resp.get("cost_usd", 0),
                    "duration_s": resp.get("duration_s", 0),
                    "summary": resp.get("summary", ""),
                    "detail": resp.get("detail", "")[:500],
                }])
                st.session_state["metric_history"] = pd.concat(
                    [st.session_state["metric_history"], new_row], ignore_index=True
                ).tail(100)
                st.rerun()

# ── KPI Cards ─────────────────────────────────────────────
hist = st.session_state["metric_history"]
total_runs = len(hist)
total_cost = hist["cost"].sum() if total_runs > 0 else 0
total_tokens = hist[["tokens_in", "tokens_out"]].sum().sum() if total_runs > 0 else 0
avg_duration = hist["duration_s"].mean() if total_runs > 0 else 0

st.subheader("📈 Aggregate Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Agent Runs", total_runs + st.session_state["run_counter"])
c2.metric("Total Tokens", f"{total_tokens:,.0f}")
c3.metric("Total Cost", f"${total_cost:.4f}")
c4.metric("Avg Duration", f"{avg_duration:.2f}s")
c5.metric("Success Rate", f"{100 * (hist['status'] == 'ok').sum() / max(total_runs, 1):.0f}%")
st.caption("Metrics tracked for this session. Runs are agent invocations via the sidebar or chat.")

# ── Charts ────────────────────────────────────────────────
if total_runs > 0:
    col_left, col_right = st.columns(2)

    with col_left:
        # Cost by agent
        cost_by_agent = hist.groupby("agent")["cost"].sum().reset_index()
        fig1 = px.bar(cost_by_agent, x="agent", y="cost", title="💰 Cost by Agent ($)",
                      color="agent", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig1, use_container_width=True)

        # Agent run count
        runs_by_agent = hist["agent"].value_counts().reset_index()
        runs_by_agent.columns = ["agent", "runs"]
        fig2 = px.pie(runs_by_agent, values="runs", names="agent", title="🔄 Runs by Agent",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # Duration by agent
        dur_by_agent = hist.groupby("agent")["duration_s"].mean().reset_index()
        fig3 = px.bar(dur_by_agent, x="agent", y="duration_s",
                      title="⏱️ Avg Duration by Agent (s)",
                      color="agent", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig3, use_container_width=True)

        # Timeline of runs
        hist_sorted = hist.sort_values("timestamp")
        fig4 = px.scatter(hist_sorted, x="timestamp", y="agent", size="tokens_in",
                          color="status", title="📡 Agent Run Timeline",
                          labels={"timestamp": "Time", "agent": "Agent"},
                          color_discrete_map={"ok": "green", "error": "red", "partial": "orange"},
                          size_max=30)
        st.plotly_chart(fig4, use_container_width=True)

    # Recent runs table
    st.divider()
    st.subheader("🕐 Recent Agent Runs")
    recent = hist.tail(10).sort_values("timestamp", ascending=False)
    display = recent[["agent", "status", "summary", "duration_s", "cost"]].copy()
    display["duration_s"] = display["duration_s"].round(2)
    display["cost"] = display["cost"].round(6)
    st.dataframe(display, use_container_width=True, hide_index=True,
                 column_config={
                     "agent": "Agent",
                     "status": st.column_config.TextColumn("Status"),
                     "summary": st.column_config.TextColumn("Summary", width="large"),
                     "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
                     "cost": st.column_config.NumberColumn("Cost ($)", format="%.6f"),
                 })
else:
    st.info("No agent runs yet. Use the sidebar to run agents and populate the dashboard.")

# ── Auto-refresh ──────────────────────────────────────────
if st.session_state.get("monitor_running"):
    time.sleep(5)
    st.rerun()

# ── Control ───────────────────────────────────────────────
st.divider()
auto_col1, auto_col2 = st.columns([1, 5])
with auto_col1:
    if st.button("🔄 Auto-refresh ON" if not st.session_state["monitor_running"] else "⏹️ Stop auto-refresh",
                 type="primary" if not st.session_state["monitor_running"] else "secondary",
                 use_container_width=True):
        st.session_state["monitor_running"] = not st.session_state["monitor_running"]
        st.rerun()

with auto_col2:
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state["metric_history"] = pd.DataFrame(columns=[
            "timestamp", "agent", "status", "tokens_in", "tokens_out", "cost", "duration_s", "summary", "detail"
        ])
        st.session_state["run_counter"] = 0
        st.rerun()