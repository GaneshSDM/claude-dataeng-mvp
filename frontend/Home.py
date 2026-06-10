import streamlit as st
from api import get
from core.deck_export import build_deck
from core.metrics import roi

st.set_page_config(page_title="DataEng Copilot", page_icon="⚡", layout="wide")
st.title("⚡ DataEng Copilot")
st.subheader("Claude runs your data engineering lifecycle. Your team reviews and steers.")

st.markdown("""
**The pitch in one line:** the same data-engineering output with a fraction of the
headcount cost — or the same team shipping multiples more. Built on **Claude
(`claude-opus-4-8`)** with a fully auditable, step-by-step event trail.

- **Profile → ETL → anomaly scan → ad-hoc SQL → executive report** — one autonomous run
- **Live or replay** — pitch-safe deterministic mode, honest labeling
- **Numbers, not vibes** — tokens, dollars, minutes vs human-baseline hours on screen
""")

latest = get("/latest-summary")
st.divider()
if latest:
    s = latest["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tool calls", s["tool_calls"])
    c2.metric("Agent minutes", s["agent_minutes"])
    c3.metric("Human-baseline hours", s["human_hours"])
    c4.metric("API cost", f"${s['total_cost_usd']}")
    st.caption("Numbers from the most recent demo run on this machine.")
    deck = build_deck(s, roi(8, 120_000, 0.5, 6))
else:
    st.info("No run yet — open **Live Demo** and press Start. Deck below uses canonical numbers.")
    deck = build_deck({"tool_calls": 14, "total_cost_usd": 1.82, "agent_minutes": 12.5,
                       "human_hours": 14.0, "speedup_x": 67.2,
                       "tokens_in": 250_000, "tokens_out": 18_000},
                      roi(8, 120_000, 0.5, 6))

st.download_button("📥 Download executive deck (PPTX)", deck,
                   file_name="dataeng-copilot-pitch.pptx",
                   mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
