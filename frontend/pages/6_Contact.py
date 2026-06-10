import streamlit as st

st.set_page_config(page_title="Contact", page_icon="✉️")
st.title("✉️ Let's talk")
st.markdown("""
**Pitch:** I run data engineering on Claude. Same productivity, reduced headcount cost —
demonstrated live, measured per step, auditable end to end.

**5 specialist agents** working together: SQL, ETL, Quality, Analytics, Documentation —
orchestrated by a smart routing layer.

**Who I want to reach:** Anthropic partnerships / solutions teams and C-level data leaders.

- 📧 **Email:** [gsundar4121@gmail.com](mailto:gsundar4121@gmail.com?subject=DataEng%20Copilot%20demo)
- 📅 **Book a demo:** calendly link goes here
- 📦 **Take-away:** download the exec deck from the Home page

*Built with the Anthropic Python SDK on `claude-opus-4-8`.*
""")
st.link_button("Request a live demo",
               "mailto:gsundar4121@gmail.com?subject=DataEng%20Copilot%20demo")