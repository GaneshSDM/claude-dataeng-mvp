import pandas as pd
import plotly.express as px
import streamlit as st
from api import get  # noqa: F401  (keeps sys.path side-effect for core imports)
from core.metrics import roi, PROGRAM_COST_YEAR1, PROGRAM_COST_ONGOING

st.set_page_config(page_title="ROI Calculator", page_icon="📈", layout="wide")
st.title("📈 ROI Calculator")
st.caption("Illustrative model. Every assumption is a slider — bring your own numbers.")

c1, c2 = st.columns([1, 2])
with c1:
    team = st.slider("Data engineering team size", 2, 50, 8)
    cost = st.slider("Avg loaded cost per engineer ($/yr)", 80_000, 250_000, 120_000, step=5_000)
    auto = st.slider("Task-hours automatable (%)", 20, 80, 50) / 100
    ramp = st.slider("Adoption ramp (months)", 1, 18, 6)
    framing = st.radio("Framing", ["Reduce headcount", "Same team, multiplied output"])

r = roi(team, cost, auto, ramp)

with c2:
    m1, m2, m3 = st.columns(3)
    if framing == "Reduce headcount":
        m1.metric("FTE-equivalent reduction", r["fte_freed"])
        headline = f"~${r['annual_savings']:,.0f}/yr run-rate savings at full ramp"
    else:
        m1.metric("Extra-capacity FTEs", r["fte_freed"])
        headline = f"~{r['fte_freed']} engineers' worth of new output, no new hires"
    m2.metric("Annual value at ramp", f"${r['annual_savings']:,.0f}")
    m3.metric("Payback month", r["payback_month"] or "—")
    st.success(headline)
    df = pd.DataFrame({"Month": range(1, 25), "Cumulative net value ($)": r["monthly_net"]})
    st.plotly_chart(px.area(df, x="Month", y="Cumulative net value ($)",
                            title="Cumulative net value, 24 months"),
                    use_container_width=True)
    st.caption(f"Program cost assumptions: ${PROGRAM_COST_YEAR1:,.0f} year 1, "
               f"${PROGRAM_COST_ONGOING:,.0f}/yr ongoing (API + enablement).")
