# DataEng Copilot — Claude for Data Engineering (Pitch MVP)

Autonomous Claude agent runs a full data-engineering lifecycle (profile → ETL →
anomaly scan → SQL answers → exec report) on a retail dataset, with per-step
cost/time metrics, an ROI calculator, and a one-click executive PPTX deck.

## Quickstart

    pip install -r requirements.txt
    python scripts/make_recording.py   # build the replay recording (no API key needed)
    .\run.ps1                          # backend :8000 + frontend :8501

Open http://127.0.0.1:8501 → **Live Demo** → Start run (replay).

## Live mode

    $env:ANTHROPIC_API_KEY = "sk-ant-..."   # then restart backend

Model: `claude-opus-4-8`. Record a fresh canonical run with `$env:RECORD = "1"`.

## Pages

| Page | What it does |
|---|---|
| Home | Pitch landing + latest run metrics + PPTX deck download |
| Live Demo | Watch the agent run (live or replay), step cards, charts, exec report, run economics |
| ROI Calculator | Slider model: FTE freed, $/yr, payback chart, headcount-vs-output framing |
| Contact | CTA + email |

## Tests

    pytest
