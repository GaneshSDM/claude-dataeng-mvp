import streamlit as st
from api import get
from core.deck_export import build_deck
from core.metrics import roi

st.set_page_config(page_title="DataEng Copilot", page_icon="⚡", layout="wide")
st.title("⚡ DataEng Copilot — Multi-Agent Platform")
st.subheader("Autonomous data engineering agents. Orchestrated. Auditable. Extensible.")

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
**One platform. Five specialist agents. Unlimited possibilities.**

Profile → Clean → Anomaly Scan → SQL Answers → Reports → Docs — all driven by
specialist AI agents working together. Built on **Claude** for the most
demanding data engineering tasks.

- 🤖 **5 agents:** SQL, ETL, Quality, Analytics, Documentation
- 🔄 **Orchestrator** routes your requests to the right agent(s)
- 🧩 **Visual pipeline builder** — compose agents into DAGs
- 💬 **Chat interface** — ask questions in natural language
- 📊 **Live monitoring** — watch agents work in real time
""")

health = get("/health")
agents = (health or {}).get("agents", [])

# ── Agent cards ───────────────────────────────────────────
st.divider()
st.subheader("🤖 Agent Fleet")
cols = st.columns(len(agents))
for i, a in enumerate(agents):
    with cols[i]:
        st.markdown(f"**{a['name'].title()}**  ")
        st.caption(a["description"][:80] + "...")

# ── Latest run metrics ────────────────────────────────────
latest = get("/latest-summary")
st.divider()
if latest:
    s = latest["summary"]
    st.subheader("📊 Latest Run Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tool calls", s.get("tool_calls", "—"))
    c2.metric("Agent minutes", s.get("agent_minutes", "—"))
    c3.metric("Human-baseline hours", s.get("human_hours", "—"))
    c4.metric("Speedup", f"{s.get('speedup_x', 0)}x")
    c5.metric("API cost", f"${s.get('total_cost_usd', 0)}")
    st.caption("Most recent demo run on this machine.")
    deck = build_deck(s, roi(8, 120_000, 0.5, 6))
else:
    st.info("No run yet. Use **Chat** or **Pipeline Builder** to get started.")
    deck = build_deck({"tool_calls": 36, "total_cost_usd": 3.50, "agent_minutes": 15.2,
                       "human_hours": 28.0, "speedup_x": 110.5,
                       "tokens_in": 500_000, "tokens_out": 36_000},
                      roi(8, 120_000, 0.5, 6))

col1, col2 = st.columns([1, 3])
with col1:
    st.download_button("📥 Download exec deck (PPTX)", deck,
                       file_name="dataeng-copilot-pitch.pptx",
                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")