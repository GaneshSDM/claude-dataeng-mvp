# DataEng Copilot — Multi-Agent Data Engineering Platform

Autonomous data engineering platform with **5 specialist agents** + orchestrator.

| Agent | Role |
|-------|------|
| **SQL Agent** | Natural language → SQL, executes queries, charts results |
| **ETL Agent** | Data profiling, cleaning, transformation, mart building |
| **Quality Agent** | Anomaly detection (5-layer), PII scanning, quality checks |
| **Analytics Agent** | Executive summaries, KPI reports, segmentation, charts |
| **Documentation Agent** | Data dictionaries, schema docs, pipeline documentation |
| **Orchestrator** | Routes requests to the right agent(s), chains multi-agent workflows |

## Quickstart

```bash
pip install -r requirements.txt
python scripts/make_recording.py   # build replay recording (optional)
.\run.ps1                          # backend :8000 + frontend :8501
```

Open **http://127.0.0.1:8501**

## Frontend Pages

| Page | URL path | What it does |
|------|----------|--------------|
| 🏠 Home | `/` | Landing, agent fleet overview, latest metrics, PPTX download |
| 💬 Chat | `/Chat` | Natural language chat → orchestrator routes to agents |
| 🔧 Pipeline Builder | `/Pipeline_Builder` | Visual DAG builder — compose agent steps, run, monitor |
| 📊 Monitoring | `/Monitoring` | Agent activity dashboard, cost/duration/success metrics |
| 📈 ROI Calculator | `/ROI_Calculator` | Slider model: FTE freed, savings, payback chart |
| ✉️ Contact | `/Contact` | CTA + email |

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | System health + agent list |
| GET | `/agents` | List all agents with capabilities |
| POST | `/agents/run` | Run a single agent (or auto-route via orchestrator) |
| POST | `/agents/pipeline` | Run an ordered multi-step pipeline |
| POST | `/jobs` | Original lifecycle job (live or replay) |
| GET | `/jobs/{id}/events` | Job events stream |

### Example: Run an agent

```bash
curl -X POST http://127.0.0.1:8000/agents/run \
  -H "Content-Type: application/json" \
  -d '{"agent": "", "input": "Show me top 5 products by revenue"}'
```

### Example: Multi-step pipeline

```bash
curl -X POST http://127.0.0.1:8000/agents/pipeline \
  -H "Content-Type: application/json" \
  -d '{"steps": [
    {"agent": "sql", "input": "Top 5 products by revenue"},
    {"agent": "analytics", "input": "Executive summary"}
  ]}'
```

## Live mode (Claude API)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# restart backend
```

## Tests

```bash
pytest
```

## Architecture

```
Frontend (Streamlit)                   Backend (FastAPI)
┌─────────┬──────┬──────────┬──────┐   ┌──────────────────────┐
│ Home /  │ Chat │ Pipeline │ Mon. │   │  Orchestrator Agent  │
│ Chat │ Mon. │          │      │   │  ┌────┬────┬────┬───┐  │
│ Builder │      │          │      │   │SQL │ETL │Qual│An.│  │
└─────────┴──────┴──────────┴──────┘   │Anal│yze │ity │Doc│  │
        ↕ REST API ↕                   └────┴────┴────┴───┘  │
┌──────────────────────────────────┐   ┌──────────────────────┘
│     DuckDB (in-memory)          │   │
│  orders · customers · products  │   │ Tool layer
└──────────────────────────────────┘   └──────────────────────┘
```

Built with the Anthropic Python SDK on `claude-opus-4-8`.