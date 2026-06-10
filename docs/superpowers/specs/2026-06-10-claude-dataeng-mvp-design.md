# DataEng Copilot MVP — Design

**Date:** 2026-06-10
**Goal:** Business-ready MVP to pitch Anthropic officials and C-level executives: Claude-powered data engineering agent that delivers the same productivity with reduced headcount (or same headcount, multiplied output).

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| MVP form | Demo app + pitch deck + landing page, combined |
| Demo scope | Full lifecycle: ingest → profile → ETL → anomaly scan → NL→SQL → exec report |
| ROI basis | Illustrative model with adjustable sliders |
| Demo engine | Hybrid: live Claude API when key present, replay mode fallback |
| Stack | Python-only — Streamlit frontend + FastAPI backend |

## Architecture

Two processes, one repo:

```
claude-dataeng-mvp/
├── backend/                 FastAPI on :8000
│   ├── main.py              routes: POST /jobs, GET /jobs/{id}, GET /jobs/{id}/events
│   ├── agent.py             Claude tool-use loop (manual loop, Anthropic Python SDK)
│   ├── tools.py             profile_data, run_sql, run_anomaly_scan, create_chart, write_report
│   ├── jobs.py              in-memory job store, background thread runner, event log
│   ├── replay.py            stream recorded JSONL transcripts as job events
│   └── anomaly_wrapper.py   wraps existing enterprise_anomaly_detector.py
├── frontend/                Streamlit on :8501
│   ├── Home.py              pitch/landing page
│   └── pages/
│       ├── 2_Live_Demo.py   run agent, render step cards
│       ├── 3_ROI_Calculator.py
│       └── 4_Contact.py
├── core/
│   ├── events.py            shared event schema (pydantic)
│   ├── metrics.py           human-baseline constants + savings math
│   └── deck_export.py       python-pptx 10-slide exec deck generator
├── data/                    sample retail CSVs (orders, customers, products)
│   └── generate_sample.py   seeds dirty rows + anomalies deterministically
├── recordings/              checked-in replay transcripts (*.jsonl)
├── tests/                   pytest
├── requirements.txt
└── run.ps1                  starts backend + frontend
```

Frontend polls `GET /jobs/{id}/events?after=N` every ~1s. No websockets.

## Agent Engine (backend/agent.py)

- **SDK:** official `anthropic` Python SDK. Client from env (`ANTHROPIC_API_KEY`). No key → live mode disabled, replay only, UI banner explains.
- **Model:** `claude-opus-4-8` (input $5/MTok, output $25/MTok — used in cost metrics).
- **Thinking:** `thinking={"type": "adaptive"}`, `output_config={"effort": "high"}`. No temperature/top_p (removed on Opus 4.8).
- **Loop:** manual agentic loop (not tool_runner) — needed for per-step event emission, token accounting, and recording. Loop until `stop_reason == "end_turn"`; append full `response.content`; match `tool_result.tool_use_id`.
- **Tools (all run locally, DuckDB on sample data):**
  - `profile_data(table)` — row counts, null %, dtypes, sample rows
  - `run_sql(query, purpose)` — executes against DuckDB; used for ETL transforms and NL→SQL answers
  - `run_anomaly_scan(table)` — calls wrapped `enterprise_anomaly_detector.py`, returns findings
  - `create_chart(spec)` — plotly figure from declarative spec; stored as job artifact
  - `write_report(markdown)` — final exec summary artifact
- **Scenario script:** one job = full lifecycle on the retail dataset. System prompt defines the mission: clean raw tables into warehouse marts, scan quality, answer 3 canned business questions, produce exec report.
- **Events:** every step emits `JobEvent {seq, ts, type, step, title, detail, sql?, artifact_id?, tokens_in, tokens_out, cost_usd, duration_s}`. Types: `job_started, step_started, claude_text, tool_call, tool_result, chart, report, job_finished, job_failed`.
- **Recording:** `RECORD=1` env writes every event to `recordings/<scenario>.jsonl` during a live run.

## Replay Mode (backend/replay.py)

- Reads checked-in JSONL, re-emits events through same job store with pacing (scaled real delays, capped ~2s/step).
- Event schema identical to live — frontend cannot tell the difference (except `mode: replay` flag shown honestly in UI).
- Default mode when no API key. Toggle in demo page sidebar.

## Metrics → Headcount Math (core/metrics.py)

- Per-step human baseline constants (illustrative, cited in UI footnote): e.g. data profiling 2h, ETL build 6h, anomaly triage 3h, ad-hoc SQL 1h each, exec report 2h.
- Run summary: total agent wall-time, total API cost, total human-equivalent hours, ratio.
- Same constants feed ROI page — one consistent story between demo and calculator.

## ROI Calculator (frontend/pages/3_ROI_Calculator.py)

- Sliders: team size (default 8), avg loaded cost ($120k), % task-hours automatable (40–60% default 50%), adoption ramp months.
- Outputs: FTE-equivalent capacity freed, $/yr saved, payback timeline chart.
- Framing toggle: "reduce headcount" vs "same team, multiplied output" — both computed from same model.
- Pure client-side math in Streamlit; no backend call.

## Pitch Page + Deck + Contact

- **Home.py:** headline, problem (data eng cost/backlog), solution (Claude agent), live numbers from latest demo run (reads job store via API; falls back to recorded numbers), "Built on Claude" positioning, CTA buttons.
- **Deck export:** button on Home → `core/deck_export.py` builds 10-slide PPTX: title, problem, solution architecture, demo walkthrough (chart images from artifacts), metrics, ROI scenario, security/ops posture, roadmap, ask, contact. Download via `st.download_button`.
- **4_Contact.py:** pitch summary, email (gsundar4121@gmail.com), placeholder calendly link, "request live demo" mailto CTA.

## Error Handling

- Live API failure → job `failed`, error event with message; UI shows one-click "switch to replay".
- SDK retries (default 2) for 429/5xx; job timeout 10 min.
- Missing dataset → generator runs automatically on backend startup if CSVs absent.

## Testing (pytest)

- Job lifecycle: create → events accumulate → finished.
- Replay/live schema parity: recorded events validate against `JobEvent` pydantic model.
- ROI math: known inputs → expected FTE/$ outputs.
- Anomaly wrapper: returns findings on seeded anomalies.
- Deck export: produces openable PPTX (python-pptx round-trip).
- Frontend: import smoke test only.

## Non-Goals (YAGNI)

- No auth, no multi-user, no DB persistence (in-memory + JSONL).
- No deployment config (local pitch first; Streamlit Cloud later if needed).
- No real company data.
