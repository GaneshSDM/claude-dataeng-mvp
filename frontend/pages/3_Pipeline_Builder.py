"""Visual Pipeline Builder — drag-and-free-form DAG composition of data engineering agents."""

import json
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from api import get, post

st.set_page_config(page_title="Pipeline Builder", page_icon="🔧", layout="wide")
st.title("🔧 Pipeline Builder")
st.caption("Compose data engineering pipelines from specialist agents. Arrange steps, chain inputs, run and monitor.")

# ── Agent catalog ─────────────────────────────────────────
health = get("/health") or {}
agent_list = health.get("agents", [])
agent_names = [a["name"].title() for a in agent_list]

# ── Pipeline state ────────────────────────────────────────
if "pipeline_steps" not in st.session_state:
    st.session_state["pipeline_steps"] = [
        {"agent": "ETL", "input": "Profile, clean, and build weekly sales mart", "enabled": True},
        {"agent": "Quality", "input": "Scan for anomalies and data quality issues", "enabled": True},
        {"agent": "Analytics", "input": "Generate executive summary with KPIs", "enabled": True},
    ]
if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = None
if "pipeline_running" not in st.session_state:
    st.session_state["pipeline_running"] = False

# ── Layout ────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📋 Pipeline Steps")

    # Step list
    to_remove = None
    for i, step in enumerate(st.session_state["pipeline_steps"]):
        with st.container(border=True):
            cols = st.columns([1, 2, 3, 1])
            with cols[0]:
                st.markdown(f"**#{i+1}**")
            with cols[1]:
                agent = st.selectbox("Agent", agent_names,
                                     index=agent_names.index(step["agent"])
                                     if step["agent"] in agent_names else 0,
                                     key=f"agent_{i}", label_visibility="collapsed")
            with cols[2]:
                inp = st.text_input("Input", step["input"],
                                    key=f"input_{i}", label_visibility="collapsed")
            with cols[3]:
                if st.button("🗑️", key=f"rm_{i}", help="Remove step"):
                    to_remove = i
            # Update
            st.session_state["pipeline_steps"][i] = {"agent": agent, "input": inp, "enabled": True}

    if to_remove is not None:
        st.session_state["pipeline_steps"].pop(to_remove)
        st.rerun()

    # Add step button
    if st.button("+ Add Step", type="secondary", use_container_width=True):
        st.session_state["pipeline_steps"].append({
            "agent": "SQL", "input": "What are the top products?", "enabled": True
        })
        st.rerun()

    # Presets
    st.divider()
    st.caption("📦 Preset Pipelines")
    presets = {
        "Full Lifecycle": [
            {"agent": "ETL", "input": "Profile, clean, build weekly sales mart", "enabled": True},
            {"agent": "Quality", "input": "Full quality scan: anomalies, PII, outliers", "enabled": True},
            {"agent": "Analytics", "input": "Executive summary with KPIs and trends", "enabled": True},
            {"agent": "Docs", "input": "Generate data dictionary and pipeline docs", "enabled": True},
        ],
        "Quick Analysis": [
            {"agent": "SQL", "input": "Top 10 products by revenue", "enabled": True},
            {"agent": "SQL", "input": "Monthly revenue trend", "enabled": True},
            {"agent": "Analytics", "input": "Executive summary", "enabled": True},
        ],
        "Quality Audit": [
            {"agent": "Quality", "input": "Full scan: anomalies, PII, outliers, quality profile", "enabled": True},
            {"agent": "Docs", "input": "Generate data dictionary", "enabled": True},
        ],
    }
    for name, steps in presets.items():
        if st.button(f"📌 {name}", use_container_width=True, type="secondary"):
            st.session_state["pipeline_steps"] = steps
            st.session_state["pipeline_results"] = None
            st.rerun()

# ── Right: Run & Results ──────────────────────────────────
with col_right:
    st.subheader("▶️ Run Pipeline")

    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True,
                        disabled=st.session_state["pipeline_running"])

    if run_btn:
        st.session_state["pipeline_results"] = None
        st.session_state["pipeline_running"] = True
        steps = [
            {"agent": s["agent"].lower(), "input": s["input"]}
            for s in st.session_state["pipeline_steps"]
            if s["enabled"]
        ]
        resp = post("/agents/pipeline", {"steps": steps})

        if resp and "results" in resp:
            st.session_state["pipeline_results"] = resp["results"]
            st.session_state["pipeline_running"] = False
        else:
            st.error("Pipeline execution failed")
            st.session_state["pipeline_running"] = False

    # Show results
    if st.session_state["pipeline_running"]:
        st.info("⏳ Pipeline running...")
    elif st.session_state["pipeline_results"]:
        results = st.session_state["pipeline_results"]
        st.success(f"Pipeline complete: {len(results)} steps")

        for i, r in enumerate(results):
            status_icon = "✅" if r.get("status") == "ok" else "❌"
            with st.expander(f"{status_icon} Step {i+1}: {r['agent'].title()}", expanded=i == 0):
                if r.get("error"):
                    st.error(r["error"])
                else:
                    st.markdown(f"**{r.get('summary', '')}**")
                    if r.get("duration_s"):
                        st.caption(f"Duration: {r['duration_s']}s")
                    if r.get("sql"):
                        st.code(r["sql"], language="sql")
                    if r.get("detail"):
                        st.markdown(r["detail"][:800])
                    # Charts
                    for art in (r.get("artifacts") or []):
                        if art.get("type") == "chart":
                            df = pd.DataFrame({
                                art.get("x_label", "x"): art["x"],
                                art.get("y_label", "y"): art["y"],
                            })
                            fn = px.line if art["chart_type"] == "line" else px.bar
                            st.plotly_chart(fn(df, x=df.columns[0], y=df.columns[1],
                                               title=art["title"]), use_container_width=True)