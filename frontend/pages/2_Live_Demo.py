import json
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from api import get, post

st.set_page_config(page_title="Live Demo", page_icon="🛠️", layout="wide")
st.title("🛠️ Live Demo — full data-engineering lifecycle")

health = get("/health") or {}
live_ok = health.get("live_available", False)

with st.sidebar:
    mode = st.radio("Engine", ["replay", "live"],
                    help="Replay = recorded run (real tool outputs, scripted narration), zero API cost. Live = real Claude API.")
    if mode == "live" and not live_ok:
        st.warning("ANTHROPIC_API_KEY not set on backend — live unavailable.")
    start = st.button("▶ Start run", type="primary",
                      disabled=(mode == "live" and not live_ok))

if start:
    r = post("/jobs", {"mode": mode, "speed": 1.0})
    if "job_id" in r:
        st.session_state["job_id"] = r["job_id"]
        st.session_state["events"] = []
    else:
        st.error(r.get("error", "failed to start"))

jid = st.session_state.get("job_id")
if not jid:
    st.info("Press **Start run** in the sidebar.")
    st.stop()

resp = get(f"/jobs/{jid}/events", params={"after": len(st.session_state["events"])})
if resp:
    st.session_state["events"].extend(resp["events"])
    status = resp["status"]
else:
    status = "unknown"

events = st.session_state["events"]
if events and events[0].get("mode") == "replay":
    st.caption("🎬 Replay mode — recorded run; tool outputs are real, narration scripted.")

for e in events:
    t = e["type"]
    if t == "job_started":
        st.success(f"Job started — {e['detail']}")
    elif t == "claude_text":
        st.chat_message("assistant").write(e["detail"])
    elif t == "tool_call":
        with st.expander(f"🔧 {e['title']}  ·  step: {e['step'] or '—'}", expanded=False):
            if e.get("sql"):
                st.code(e["sql"], language="sql")
            else:
                st.code(e["detail"], language="json")
    elif t == "tool_result":
        with st.expander(f"📄 result · {e['title']}", expanded=False):
            st.code(e["detail"][:1500], language="json")
    elif t == "chart" and e.get("artifact"):
        a = e["artifact"]
        df = pd.DataFrame({a.get("x_label") or "x": a["x"], a.get("y_label") or "y": a["y"]})
        fig = (px.line if a["chart_type"] == "line" else px.bar)(
            df, x=df.columns[0], y=df.columns[1], title=a["title"])
        st.plotly_chart(fig, use_container_width=True)
    elif t == "report" and e.get("artifact"):
        st.divider()
        st.markdown(e["artifact"]["markdown"])
    elif t == "job_finished":
        s = json.loads(e["detail"])
        st.divider()
        st.subheader("Run economics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tool calls", s["tool_calls"])
        c2.metric("Agent minutes", s["agent_minutes"])
        c3.metric("Human hours replaced", s["human_hours"])
        c4.metric("Speedup", f"{s['speedup_x']}x")
        c5.metric("API cost", f"${s['total_cost_usd']}")
        st.caption("Human-baseline hours are illustrative industry-typical estimates per task type.")
    elif t == "job_failed":
        st.error(f"Run failed: {e['detail']}")
        if st.button("Switch to replay"):
            r = post("/jobs", {"mode": "replay", "speed": 1.0})
            st.session_state["job_id"] = r.get("job_id")
            st.session_state["events"] = []
            st.rerun()

if status == "running":
    time.sleep(1.0)
    st.rerun()
