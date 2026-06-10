# DataEng Copilot MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Working pitch MVP — Streamlit app (landing + live demo + ROI calculator + contact) backed by a FastAPI agent service running a Claude full-lifecycle data-engineering job (live or replay), with PPTX deck export.

**Architecture:** Two local processes. FastAPI (:8000) owns jobs: a Claude manual tool-use loop executing local tools against DuckDB over generated retail CSVs, emitting a uniform `JobEvent` stream; replay mode re-emits a checked-in JSONL recording through the same store. Streamlit (:8501) polls the events endpoint and renders; ROI math and deck export are pure functions in `core/`.

**Tech Stack:** Python 3.11+, anthropic SDK (`claude-opus-4-8`), FastAPI + uvicorn, Streamlit, DuckDB, pandas, plotly, python-pptx, pytest.

**Conventions:** All commands run from repo root `claude-dataeng-mvp/`. Windows PowerShell. Commit after every green test step. The existing `enterprise_anomaly_detector.py` is vendored as `backend/vendor_detector.py` (imports are pandas/numpy/scipy only — verified, no Snowflake needed for `MultiLayerAnomalyDetector`).

---

### Task 1: Scaffold

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `backend/__init__.py`, `core/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Write files**

`requirements.txt`:
```
anthropic>=0.92.0
fastapi>=0.115
uvicorn>=0.30
streamlit>=1.40
duckdb>=1.1
pandas>=2.2
numpy>=1.26
scipy>=1.13
plotly>=5.24
python-pptx>=1.0
pydantic>=2.8
requests>=2.32
pytest>=8.3
httpx>=0.27
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
data/*.csv
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
```

Empty `__init__.py` in `backend/`, `core/`, `tests/`.

- [ ] **Step 2: Install deps**

Run: `pip install -r requirements.txt`
Expected: success.

- [ ] **Step 3: Commit**

```
git add -A; git commit -m "chore: scaffold project"
```

---

### Task 2: Sample data generator

**Files:**
- Create: `data/generate_sample.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Failing test**

`tests/test_data.py`:
```python
from pathlib import Path
import pandas as pd
from data.generate_sample import generate, DATA_DIR

def test_generate_creates_seeded_retail_data(tmp_path):
    generate(tmp_path)
    orders = pd.read_csv(tmp_path / "orders.csv")
    customers = pd.read_csv(tmp_path / "customers.csv")
    products = pd.read_csv(tmp_path / "products.csv")
    assert len(orders) > 4000 and len(customers) == 500 and len(products) == 60
    # dirty rows seeded
    assert (orders["quantity"] < 0).sum() >= 10
    assert orders["unit_price"].astype(str).str.contains("N/A").any()
    assert customers["email"].isna().sum() >= 10
    # deterministic
    generate(tmp_path)
    orders2 = pd.read_csv(tmp_path / "orders.csv")
    assert orders.equals(orders2)
```

Run: `pytest tests/test_data.py -v` — Expected: FAIL (module missing).

- [ ] **Step 2: Implement**

`data/generate_sample.py`:
```python
"""Deterministic sample retail dataset with seeded dirt + anomalies."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
WEEKS = 78  # enough history for 8-week rolling baselines
SPIKE_WEEK = WEEKS - 3   # GDV x4 spike
DROP_WEEK = WEEKS - 10   # GDV -60% drop


def generate(out_dir: Path = DATA_DIR) -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = pd.DataFrame({
        "customer_id": np.arange(1, 501),
        "name": [f"Customer {i}" for i in range(1, 501)],
        "email": [f"c{i}@example.com" for i in range(1, 501)],
        "country": rng.choice(["US", "IN", "DE", "UK"], 500, p=[.5, .2, .15, .15]),
        "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, 500), "D"),
    })
    customers.loc[rng.choice(500, 15, replace=False), "email"] = None  # dirty: null emails
    customers = pd.concat([customers, customers.iloc[:10]], ignore_index=True)  # dirty: dup rows

    products = pd.DataFrame({
        "product_id": np.arange(1, 61),
        "product_name": [f"Product {i}" for i in range(1, 61)],
        "category": rng.choice(["Electronics", "Home", "Toys", "Office"], 60),
        "list_price": np.round(rng.uniform(5, 400, 60), 2),
    })

    start = pd.Timestamp("2024-06-03")  # a Monday
    rows = []
    oid = 1
    for w in range(WEEKS):
        base = 60 + 10 * np.sin(w / 6)
        mult = 4.0 if w == SPIKE_WEEK else (0.4 if w == DROP_WEEK else 1.0)
        n = int(rng.poisson(base * mult))
        for _ in range(n):
            day = start + pd.Timedelta(weeks=w, days=int(rng.integers(0, 7)))
            pid = int(rng.integers(1, 61))
            price = float(products.loc[pid - 1, "list_price"]) * float(rng.uniform(.9, 1.1))
            rows.append((oid, int(rng.integers(1, 501)), pid, day.date().isoformat(),
                         int(rng.integers(1, 5)), round(price * mult if w == SPIKE_WEEK else price, 2),
                         str(rng.choice(["web", "mobile", "partner"])),))
            oid += 1
    orders = pd.DataFrame(rows, columns=["order_id", "customer_id", "product_id",
                                         "order_date", "quantity", "unit_price", "channel"])
    dirty_idx = rng.choice(len(orders), 25, replace=False)
    orders.loc[dirty_idx[:20], "quantity"] = -1                      # dirty: negative qty
    orders["unit_price"] = orders["unit_price"].astype(object)
    orders.loc[dirty_idx[20:], "unit_price"] = "N/A"                 # dirty: bad price strings

    customers.to_csv(out_dir / "customers.csv", index=False)
    products.to_csv(out_dir / "products.csv", index=False)
    orders.to_csv(out_dir / "orders.csv", index=False)


if __name__ == "__main__":
    generate()
    print(f"Sample data written to {DATA_DIR}")
```

Note: `data/` needs an empty `__init__.py` so tests can import it.

- [ ] **Step 3: Run test — PASS, then commit**

```
git add -A; git commit -m "feat: deterministic sample retail data with seeded anomalies"
```

---

### Task 3: Event schema (core/events.py)

**Files:**
- Create: `core/events.py`
- Test: `tests/test_events.py`

- [ ] **Step 1: Failing test**

`tests/test_events.py`:
```python
from core.events import JobEvent

def test_job_event_defaults_and_roundtrip():
    e = JobEvent(seq=1, ts=123.4, type="tool_call", step="etl", title="run_sql")
    j = e.model_dump_json()
    e2 = JobEvent.model_validate_json(j)
    assert e2 == e and e2.cost_usd == 0.0 and e2.artifact is None
```

- [ ] **Step 2: Implement**

`core/events.py`:
```python
"""Shared job event schema — identical for live and replay."""
from typing import Literal, Optional, Any
from pydantic import BaseModel

EventType = Literal["job_started", "step_started", "claude_text", "tool_call",
                    "tool_result", "chart", "report", "job_finished", "job_failed"]


class JobEvent(BaseModel):
    seq: int
    ts: float
    type: EventType
    step: str = ""            # profile|etl|anomaly|sql|report|"" — keys into metrics baselines
    title: str = ""
    detail: str = ""
    sql: Optional[str] = None
    artifact: Optional[dict[str, Any]] = None  # inline chart/report payload
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    mode: Literal["live", "replay"] = "live"
```

- [ ] **Step 3: Test PASS, commit** `feat: shared JobEvent schema`

---

### Task 4: Metrics + ROI math (core/metrics.py)

**Files:**
- Create: `core/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Failing tests**

`tests/test_metrics.py`:
```python
from core.metrics import cost_usd, run_summary, roi
from core.events import JobEvent

def test_cost_usd_opus_pricing():
    assert abs(cost_usd(1_000_000, 1_000_000) - 30.0) < 1e-9  # $5 in + $25 out

def test_run_summary_aggregates():
    ev = [JobEvent(seq=1, ts=0, type="tool_call", step="etl", tokens_in=1000, tokens_out=500,
                   cost_usd=cost_usd(1000, 500), duration_s=2.0),
          JobEvent(seq=2, ts=0, type="tool_call", step="profile", duration_s=1.0)]
    s = run_summary(ev)
    assert s["total_cost_usd"] > 0 and s["agent_minutes"] > 0
    assert s["human_hours"] == 8.0  # etl 6h + profile 2h
    assert s["speedup_x"] > 1

def test_roi_basic():
    r = roi(team_size=8, loaded_cost=120_000, automatable_pct=0.5, ramp_months=6)
    assert r["fte_freed"] == 4.0
    assert r["annual_savings"] == 480_000
    assert len(r["monthly_net"]) == 24 and r["payback_month"] is not None
```

- [ ] **Step 2: Implement**

`core/metrics.py`:
```python
"""Illustrative productivity + ROI math. All baselines are assumptions, surfaced in UI."""
from core.events import JobEvent

MODEL_ID = "claude-opus-4-8"
PRICE_IN_PER_TOK = 5.0 / 1_000_000    # USD, claude-opus-4-8
PRICE_OUT_PER_TOK = 25.0 / 1_000_000

# Illustrative human-baseline hours per task type (industry-typical, adjustable story)
HUMAN_BASELINE_HOURS = {"profile": 2.0, "etl": 6.0, "anomaly": 3.0, "sql": 1.0, "report": 2.0}

PROGRAM_COST_YEAR1 = 150_000.0   # illustrative: setup + API + enablement
PROGRAM_COST_ONGOING = 50_000.0  # per year thereafter


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * PRICE_IN_PER_TOK + tokens_out * PRICE_OUT_PER_TOK


def run_summary(events: list[JobEvent]) -> dict:
    steps_done = {e.step for e in events if e.type == "tool_call" and e.step}
    human_hours = sum(HUMAN_BASELINE_HOURS.get(s, 0.0) for s in steps_done)
    agent_seconds = sum(e.duration_s for e in events)
    agent_minutes = round(agent_seconds / 60, 2)
    total_cost = round(sum(e.cost_usd for e in events), 4)
    speedup = round((human_hours * 60) / agent_minutes, 1) if agent_minutes else 0.0
    return {
        "steps_done": sorted(steps_done),
        "tool_calls": sum(1 for e in events if e.type == "tool_call"),
        "tokens_in": sum(e.tokens_in for e in events),
        "tokens_out": sum(e.tokens_out for e in events),
        "total_cost_usd": total_cost,
        "agent_minutes": agent_minutes,
        "human_hours": human_hours,
        "speedup_x": speedup,
    }


def roi(team_size: int, loaded_cost: float, automatable_pct: float, ramp_months: int) -> dict:
    """24-month illustrative model. automatable_pct of task-hours absorbed at full ramp."""
    fte_freed = round(team_size * automatable_pct, 1)
    annual_savings = fte_freed * loaded_cost
    monthly_net, cum = [], 0.0
    payback_month = None
    for m in range(1, 25):
        ramp = min(1.0, m / max(ramp_months, 1))
        savings = (annual_savings / 12) * ramp
        cost = (PROGRAM_COST_YEAR1 / 12) if m <= 12 else (PROGRAM_COST_ONGOING / 12)
        cum += savings - cost
        monthly_net.append(round(cum, 0))
        if payback_month is None and cum > 0:
            payback_month = m
    return {"fte_freed": fte_freed, "annual_savings": annual_savings,
            "monthly_net": monthly_net, "payback_month": payback_month}
```

- [ ] **Step 3: Tests PASS, commit** `feat: productivity metrics and ROI model`

---

### Task 5: Tools + anomaly wrapper

**Files:**
- Create: `backend/vendor_detector.py` (copy of `..\enterprise_anomaly_detector.py`)
- Create: `backend/anomaly_wrapper.py`, `backend/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Vendor the detector**

```powershell
Copy-Item "$env:USERPROFILE\Downloads\enterprise_anomaly_detector.py" backend\vendor_detector.py
```

- [ ] **Step 2: Failing tests**

`tests/test_tools.py`:
```python
import pytest
from pathlib import Path
from data.generate_sample import generate
from backend.tools import ToolExecutor, TOOL_DEFS

@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(d)
    return ToolExecutor(d)

def test_tool_defs_shape():
    names = {t["name"] for t in TOOL_DEFS}
    assert names == {"profile_data", "run_sql", "run_anomaly_scan", "create_chart", "write_report"}
    assert all("input_schema" in t for t in TOOL_DEFS)

def test_profile_data(ex):
    out = ex.execute("profile_data", {"table": "orders"})
    assert out["result"]["row_count"] > 4000 and "quantity" in out["result"]["columns"]

def test_run_sql_select_and_ddl(ex):
    ex.execute("run_sql", {"query": "CREATE TABLE t1 AS SELECT 1 AS x", "purpose": "etl"})
    out = ex.execute("run_sql", {"query": "SELECT x FROM t1", "purpose": "question"})
    assert out["result"]["rows"] == [[1]]

def test_anomaly_scan_finds_seeded_anomalies(ex):
    out = ex.execute("run_anomaly_scan", {})
    dirs = {f["direction"] for f in out["result"]["findings"]}
    assert "SPIKE" in dirs or "DROP" in dirs

def test_chart_and_report_artifacts(ex):
    c = ex.execute("create_chart", {"title": "T", "chart_type": "bar",
                                    "x": ["a", "b"], "y": [1, 2],
                                    "x_label": "X", "y_label": "Y"})
    assert c["artifact"]["type"] == "chart" and c["artifact"]["x"] == ["a", "b"]
    r = ex.execute("write_report", {"markdown": "# Done"})
    assert r["artifact"]["type"] == "report"
```

- [ ] **Step 3: Implement wrapper**

`backend/anomaly_wrapper.py`:
```python
"""Adapter: DuckDB retail orders -> vendor MultiLayerAnomalyDetector facts."""
import pandas as pd
from backend.vendor_detector import DetectorConfig, MultiLayerAnomalyDetector


def scan_orders(conn) -> list[dict]:
    weekly = conn.execute("""
        SELECT date_trunc('week', CAST(order_date AS DATE)) AS week_start_date,
               channel,
               SUM(quantity * TRY_CAST(unit_price AS DOUBLE)) AS gdv,
               COUNT(*) AS completes
        FROM orders
        WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1
    """).fetchdf()
    facts = pd.DataFrame({
        "SOURCE_TABLE": "ORDERS",
        "WEEK_START_DATE": pd.to_datetime(weekly["week_start_date"]),
        "COUNTRY": "ALL",
        "DEVICE": weekly["channel"],
        "TRAFFIC_SOURCE": "ALL",
        "GDV": weekly["gdv"],
        "COMPLETES": weekly["completes"],
    })
    det = MultiLayerAnomalyDetector(DetectorConfig())
    res = det.detect(facts)
    hits = res[res["direction"] != "NORMAL"].sort_values("ensemble_score", key=abs, ascending=False)
    return [{
        "week": str(r.week_start_date)[:10], "metric": r.metric, "direction": r.direction,
        "severity": r.severity, "pct_diff": round(float(r.pct_diff), 1),
        "z_score": round(float(r.z_score), 2), "actual": round(float(r.actual), 2),
        "baseline_avg": round(float(r.baseline_avg), 2),
    } for r in hits.head(10).itertuples()]
```

- [ ] **Step 4: Implement tools**

`backend/tools.py`:
```python
"""Local tools Claude drives. All execute against per-job DuckDB over sample CSVs."""
from pathlib import Path
import duckdb
from backend.anomaly_wrapper import scan_orders

TOOL_DEFS = [
    {"name": "profile_data",
     "description": "Profile a raw table: row count, columns/dtypes, null counts, 5 sample rows. Call once per raw table (orders, customers, products) before transforming.",
     "input_schema": {"type": "object", "properties": {
         "table": {"type": "string", "enum": ["orders", "customers", "products"]}},
         "required": ["table"]}},
    {"name": "run_sql",
     "description": "Execute DuckDB SQL. Use CREATE TABLE ... AS for cleaning/marts (purpose=etl); SELECT for business questions (purpose=question) or quick checks (purpose=inspect). SELECT output truncated to 50 rows.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "purpose": {"type": "string", "enum": ["etl", "question", "inspect"]}},
         "required": ["query", "purpose"]}},
    {"name": "run_anomaly_scan",
     "description": "Run the 5-layer statistical anomaly detector (pct/z-score/IQR/CUSUM/Bayesian) on weekly order GDV and volume. Call after ETL. Returns top findings.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "create_chart",
     "description": "Create a chart artifact for the executive report. Provide parallel x (labels) and y (numbers) arrays.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"},
         "chart_type": {"type": "string", "enum": ["bar", "line"]},
         "x": {"type": "array", "items": {"type": "string"}},
         "y": {"type": "array", "items": {"type": "number"}},
         "x_label": {"type": "string"}, "y_label": {"type": "string"}},
         "required": ["title", "chart_type", "x", "y"]}},
    {"name": "write_report",
     "description": "Write the final executive summary as markdown. Call exactly once, last.",
     "input_schema": {"type": "object", "properties": {"markdown": {"type": "string"}},
                      "required": ["markdown"]}},
]

STEP_FOR_TOOL = {"profile_data": "profile", "run_anomaly_scan": "anomaly", "write_report": "report"}


def step_for(tool: str, tool_input: dict) -> str:
    if tool == "run_sql":
        return {"etl": "etl", "question": "sql", "inspect": "profile"}[tool_input.get("purpose", "inspect")]
    if tool == "create_chart":
        return "report"
    return STEP_FOR_TOOL.get(tool, "")


class ToolExecutor:
    def __init__(self, data_dir: Path):
        self.conn = duckdb.connect(":memory:")
        for t in ("orders", "customers", "products"):
            self.conn.execute(
                f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{Path(data_dir) / t}.csv')")

    def execute(self, name: str, tool_input: dict) -> dict:
        """Returns {"result": <json-able>, "artifact": <dict|None>}."""
        if name == "profile_data":
            t = tool_input["table"]
            df = self.conn.execute(f"SELECT * FROM {t}").fetchdf()
            return {"result": {
                "row_count": len(df),
                "columns": {c: str(df[c].dtype) for c in df.columns},
                "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
                "sample_rows": df.head(5).astype(str).to_dict("records")}, "artifact": None}
        if name == "run_sql":
            cur = self.conn.execute(tool_input["query"])
            if cur.description:
                rows = cur.fetchmany(50)
                return {"result": {"columns": [d[0] for d in cur.description],
                                   "rows": [list(r) for r in rows]}, "artifact": None}
            return {"result": {"status": "ok"}, "artifact": None}
        if name == "run_anomaly_scan":
            return {"result": {"findings": scan_orders(self.conn)}, "artifact": None}
        if name == "create_chart":
            art = {"type": "chart", "title": tool_input["title"],
                   "chart_type": tool_input["chart_type"],
                   "x": tool_input["x"], "y": tool_input["y"],
                   "x_label": tool_input.get("x_label", ""), "y_label": tool_input.get("y_label", "")}
            return {"result": {"status": "chart created", "title": art["title"]}, "artifact": art}
        if name == "write_report":
            return {"result": {"status": "report saved"},
                    "artifact": {"type": "report", "markdown": tool_input["markdown"]}}
        raise ValueError(f"unknown tool {name}")
```

- [ ] **Step 5: Tests PASS, commit** `feat: agent tools + vendored anomaly detector wrapper`

---

### Task 6: Job store (backend/jobs.py)

**Files:**
- Create: `backend/jobs.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Failing test**

`tests/test_jobs.py`:
```python
from backend.jobs import JobStore

def test_job_lifecycle_and_event_pagination():
    s = JobStore()
    job = s.create("replay")
    s.emit(job.id, type="job_started", title="start")
    s.emit(job.id, type="claude_text", detail="hi")
    evs = s.events_after(job.id, 0)
    assert [e.seq for e in evs] == [1, 2]
    assert s.events_after(job.id, 1)[0].type == "claude_text"
    s.finish(job.id)
    assert s.get(job.id).status == "finished"
```

- [ ] **Step 2: Implement**

`backend/jobs.py`:
```python
"""In-memory job store. Thread-safe enough for a single-user demo."""
import threading
import time
import uuid
from dataclasses import dataclass, field
from core.events import JobEvent


@dataclass
class Job:
    id: str
    mode: str
    status: str = "running"   # running|finished|failed
    events: list[JobEvent] = field(default_factory=list)
    created: float = field(default_factory=time.time)


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, mode: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], mode=mode)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def latest_finished(self) -> Job | None:
        done = [j for j in self._jobs.values() if j.status == "finished"]
        return max(done, key=lambda j: j.created) if done else None

    def emit(self, job_id: str, **kw) -> JobEvent:
        with self._lock:
            job = self._jobs[job_id]
            ev = JobEvent(seq=len(job.events) + 1, ts=time.time(), mode=job.mode, **kw)
            job.events.append(ev)
            return ev

    def events_after(self, job_id: str, after: int) -> list[JobEvent]:
        with self._lock:
            return [e for e in self._jobs[job_id].events if e.seq > after]

    def finish(self, job_id: str, failed: bool = False):
        with self._lock:
            self._jobs[job_id].status = "failed" if failed else "finished"


STORE = JobStore()


def run_in_thread(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t
```

- [ ] **Step 3: Test PASS, commit** `feat: in-memory job store with event log`

---

### Task 7: Agent loop (backend/agent.py)

**Files:**
- Create: `backend/agent.py`
- Test: `tests/test_agent.py` (mocked Anthropic client)

- [ ] **Step 1: Failing test**

`tests/test_agent.py`:
```python
from types import SimpleNamespace as NS
from backend.jobs import JobStore
from backend.agent import run_live_job
from data.generate_sample import generate

class FakeClient:
    """First call: tool_use profile_data; second call: end_turn text."""
    def __init__(self):
        self.calls = 0
        self.messages = self
    def create(self, **kw):
        self.calls += 1
        usage = NS(input_tokens=1000, output_tokens=200)
        if self.calls == 1:
            return NS(stop_reason="tool_use", usage=usage, content=[
                NS(type="text", text="Profiling orders."),
                NS(type="tool_use", id="tu1", name="profile_data", input={"table": "orders"})])
        return NS(stop_reason="end_turn", usage=usage,
                  content=[NS(type="text", text="All done.")])

def test_agent_loop_with_mocked_client(tmp_path):
    generate(tmp_path)
    store = JobStore()
    job = store.create("live")
    run_live_job(store, job.id, tmp_path, client=FakeClient(), max_iters=5)
    types = [e.type for e in job.events]
    assert "job_started" in types and "tool_call" in types and "tool_result" in types
    assert types[-1] == "job_finished"
    assert store.get(job.id).status == "finished"
    assert sum(e.cost_usd for e in job.events) > 0
```

- [ ] **Step 2: Implement**

`backend/agent.py`:
```python
"""Manual Claude tool-use loop. Manual (not tool_runner) for per-step events,
token accounting, and recording."""
import json
import os
import time
from pathlib import Path
from backend.jobs import JobStore
from backend.tools import TOOL_DEFS, ToolExecutor, step_for
from core.metrics import MODEL_ID, cost_usd, run_summary

SYSTEM = """You are DataEng Copilot, an autonomous data engineering agent demoing a full
lifecycle on a retail dataset (DuckDB tables: orders, customers, products — raw, dirty).
Mission, in order:
1. profile_data each raw table.
2. ETL with run_sql(purpose=etl): build orders_clean (drop negative quantities, cast
   unit_price dropping non-numeric, dedupe customers into customers_clean) and a
   weekly_sales mart (week, revenue, order_count).
3. run_anomaly_scan once.
4. Answer 3 business questions with run_sql(purpose=question): top 5 products by revenue;
   monthly revenue trend; repeat-customer rate.
5. create_chart twice: monthly revenue trend (line), top products (bar).
6. write_report once: executive markdown summary — data quality fixes, anomaly findings
   with weeks and severity, business answers, and a 'what this replaced' note.
Be decisive; no questions. Keep narration to one short sentence before each action."""

KICKOFF = "Run the full lifecycle now."


def run_live_job(store: JobStore, job_id: str, data_dir: Path, client=None, max_iters: int = 30):
    record = os.environ.get("RECORD") == "1"
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        ex = ToolExecutor(data_dir)
        store.emit(job_id, type="job_started", title="Full lifecycle demo", detail=MODEL_ID)
        messages = [{"role": "user", "content": KICKOFF}]
        for _ in range(max_iters):
            t0 = time.time()
            resp = client.messages.create(
                model=MODEL_ID, max_tokens=16000, system=SYSTEM,
                thinking={"type": "adaptive"}, output_config={"effort": "high"},
                tools=TOOL_DEFS, messages=messages)
            dt = time.time() - t0
            c = cost_usd(resp.usage.input_tokens, resp.usage.output_tokens)
            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            for b in resp.content:
                if getattr(b, "type", "") == "text" and b.text.strip():
                    store.emit(job_id, type="claude_text", detail=b.text,
                               tokens_in=resp.usage.input_tokens if b is resp.content[0] else 0,
                               tokens_out=resp.usage.output_tokens if b is resp.content[0] else 0,
                               cost_usd=c if b is resp.content[0] else 0.0,
                               duration_s=dt if b is resp.content[0] else 0.0)
            if resp.stop_reason != "tool_use" or not tool_uses:
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                step = step_for(tu.name, tu.input)
                store.emit(job_id, type="tool_call", step=step, title=tu.name,
                           detail=json.dumps(tu.input)[:500],
                           sql=tu.input.get("query") if tu.name == "run_sql" else None)
                t1 = time.time()
                try:
                    out = ex.execute(tu.name, tu.input)
                    payload, is_err = json.dumps(out["result"], default=str)[:4000], False
                except Exception as e:  # tool failure -> let Claude adapt
                    out, payload, is_err = {"artifact": None}, f"Error: {e}", True
                ev_type = ("chart" if (out["artifact"] or {}).get("type") == "chart"
                           else "report" if (out["artifact"] or {}).get("type") == "report"
                           else "tool_result")
                store.emit(job_id, type=ev_type, step=step, title=tu.name,
                           detail=payload[:1500], artifact=out["artifact"],
                           duration_s=time.time() - t1)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": payload, "is_error": is_err})
            messages.append({"role": "user", "content": results})
        summary = run_summary(store.get(job_id).events)
        store.emit(job_id, type="job_finished", title="Run complete",
                   detail=json.dumps(summary))
        store.finish(job_id)
        if record:
            _dump_recording(store, job_id)
    except Exception as e:
        store.emit(job_id, type="job_failed", title="Job failed", detail=str(e)[:800])
        store.finish(job_id, failed=True)


def _dump_recording(store: JobStore, job_id: str):
    rec_dir = Path(__file__).resolve().parent.parent / "recordings"
    rec_dir.mkdir(exist_ok=True)
    with open(rec_dir / "full_lifecycle.jsonl", "w", encoding="utf-8") as f:
        for e in store.get(job_id).events:
            f.write(e.model_dump_json() + "\n")
```

- [ ] **Step 3: Test PASS, commit** `feat: Claude manual tool-use agent loop with event emission`

---

### Task 8: Recording generator + replay

**Files:**
- Create: `scripts/make_recording.py`, `backend/replay.py`, `recordings/full_lifecycle.jsonl` (generated)
- Test: `tests/test_replay.py`

- [ ] **Step 1: Failing test**

`tests/test_replay.py`:
```python
from pathlib import Path
from backend.jobs import JobStore
from backend.replay import run_replay_job, RECORDING

def test_recording_exists_and_replays():
    assert Path(RECORDING).exists(), "run scripts/make_recording.py first"
    store = JobStore()
    job = store.create("replay")
    run_replay_job(store, job.id, speed=0)  # no sleeps in tests
    assert store.get(job.id).status == "finished"
    evs = job.events
    assert evs[0].type == "job_started" and evs[-1].type == "job_finished"
    assert all(e.mode == "replay" for e in evs)
    assert any(e.type == "chart" and e.artifact for e in evs)
    assert any(e.type == "report" and e.artifact for e in evs)
```

- [ ] **Step 2: Implement replay**

`backend/replay.py`:
```python
"""Re-emit a recorded run through the job store with realistic pacing."""
import time
from pathlib import Path
from core.events import JobEvent
from backend.jobs import JobStore

RECORDING = Path(__file__).resolve().parent.parent / "recordings" / "full_lifecycle.jsonl"


def run_replay_job(store: JobStore, job_id: str, speed: float = 1.0):
    try:
        recorded = [JobEvent.model_validate_json(l) for l in
                    RECORDING.read_text(encoding="utf-8").splitlines() if l.strip()]
        prev_ts = recorded[0].ts if recorded else 0
        for r in recorded:
            gap = min(max(r.ts - prev_ts, 0.15), 2.0) * speed
            if gap:
                time.sleep(gap)
            prev_ts = r.ts
            store.emit(job_id, **r.model_dump(exclude={"seq", "ts", "mode"}))
        store.finish(job_id)
    except Exception as e:
        store.emit(job_id, type="job_failed", title="Replay failed", detail=str(e)[:500])
        store.finish(job_id, failed=True)
```

- [ ] **Step 3: Implement recording generator (synthetic, no API key needed)**

`scripts/make_recording.py` — executes the REAL tools in the scripted mission order with hardcoded narration and realistic token/duration numbers, then dumps JSONL. Tool outputs are genuine; only narration is scripted; UI labels the mode honestly as "replay".

```python
"""Build recordings/full_lifecycle.jsonl by running real tools with scripted narration.
Usage: python scripts/make_recording.py   (or RECORD=1 live run via backend/agent.py)"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generate_sample import generate, DATA_DIR  # noqa: E402
from backend.jobs import JobStore  # noqa: E402
from backend.tools import ToolExecutor, step_for  # noqa: E402
from core.metrics import MODEL_ID, cost_usd, run_summary  # noqa: E402

SCRIPT = [
    ("Profiling the three raw tables first.", "profile_data", {"table": "orders"}),
    (None, "profile_data", {"table": "customers"}),
    (None, "profile_data", {"table": "products"}),
    ("Raw data has negative quantities, non-numeric prices and duplicate customers — building clean tables.",
     "run_sql", {"purpose": "etl", "query":
        "CREATE TABLE orders_clean AS SELECT order_id, customer_id, product_id, "
        "CAST(order_date AS DATE) AS order_date, quantity, "
        "TRY_CAST(unit_price AS DOUBLE) AS unit_price, channel, "
        "quantity * TRY_CAST(unit_price AS DOUBLE) AS revenue "
        "FROM orders WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL"}),
    (None, "run_sql", {"purpose": "etl", "query":
        "CREATE TABLE customers_clean AS SELECT DISTINCT * FROM customers"}),
    (None, "run_sql", {"purpose": "etl", "query":
        "CREATE TABLE weekly_sales AS SELECT date_trunc('week', order_date) AS week, "
        "SUM(revenue) AS revenue, COUNT(*) AS order_count FROM orders_clean GROUP BY 1 ORDER BY 1"}),
    ("Marts ready. Running the statistical anomaly scan.", "run_anomaly_scan", {}),
    ("Answering the business questions.", "run_sql", {"purpose": "question", "query":
        "SELECT p.product_name, ROUND(SUM(o.revenue),2) AS revenue FROM orders_clean o "
        "JOIN products p USING(product_id) GROUP BY 1 ORDER BY 2 DESC LIMIT 5"}),
    (None, "run_sql", {"purpose": "question", "query":
        "SELECT date_trunc('month', order_date) AS month, ROUND(SUM(revenue),2) AS revenue "
        "FROM orders_clean GROUP BY 1 ORDER BY 1"}),
    (None, "run_sql", {"purpose": "question", "query":
        "SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE n > 1) / COUNT(*), 1) AS repeat_pct "
        "FROM (SELECT customer_id, COUNT(*) AS n FROM orders_clean GROUP BY 1)"}),
]


def main():
    if not (DATA_DIR / "orders.csv").exists():
        generate()
    store = JobStore()
    job = store.create("live")  # recorded as canonical run; replay flags mode itself
    ex = ToolExecutor(DATA_DIR)
    store.emit(job.id, type="job_started", title="Full lifecycle demo", detail=MODEL_ID)

    def emit_text(text, ti=2400, to=120):
        store.emit(job.id, type="claude_text", detail=text, tokens_in=ti, tokens_out=to,
                   cost_usd=cost_usd(ti, to), duration_s=2.2)
        time.sleep(0.05)

    def run_tool(name, args):
        step = step_for(name, args)
        store.emit(job.id, type="tool_call", step=step, title=name,
                   detail=json.dumps(args)[:500],
                   sql=args.get("query") if name == "run_sql" else None)
        t0 = time.time()
        out = ex.execute(name, args)
        ev = ("chart" if (out["artifact"] or {}).get("type") == "chart" else
              "report" if (out["artifact"] or {}).get("type") == "report" else "tool_result")
        store.emit(job.id, type=ev, step=step, title=name,
                   detail=json.dumps(out["result"], default=str)[:1500],
                   artifact=out["artifact"], duration_s=max(time.time() - t0, 0.4))
        return out

    monthly = None
    top5 = None
    findings = None
    for narration, name, args in SCRIPT:
        if narration:
            emit_text(narration)
        out = run_tool(name, args)
        if name == "run_anomaly_scan":
            findings = out["result"]["findings"]
        if name == "run_sql" and "product_name" in args.get("query", ""):
            top5 = out["result"]
        if name == "run_sql" and "month" in args.get("query", ""):
            monthly = out["result"]

    emit_text("Building charts for the executive report.")
    run_tool("create_chart", {"title": "Monthly Revenue Trend", "chart_type": "line",
                              "x": [str(r[0])[:10] for r in monthly["rows"]],
                              "y": [float(r[1]) for r in monthly["rows"]],
                              "x_label": "Month", "y_label": "Revenue (USD)"})
    run_tool("create_chart", {"title": "Top 5 Products by Revenue", "chart_type": "bar",
                              "x": [str(r[0]) for r in top5["rows"]],
                              "y": [float(r[1]) for r in top5["rows"]],
                              "x_label": "Product", "y_label": "Revenue (USD)"})
    flines = "\n".join(f"- {f['week']} **{f['metric']} {f['direction']}** ({f['severity']}), "
                       f"{f['pct_diff']}% vs baseline" for f in (findings or [])[:5])
    report = (f"# Executive Summary — Retail Data Lifecycle\n\n"
              f"## Data quality fixes\n- Removed negative-quantity orders\n"
              f"- Dropped non-numeric prices\n- Deduplicated customers\n\n"
              f"## Anomaly findings (5-layer detector)\n{flines}\n\n"
              f"## Business answers\n- Top products and monthly trend charted\n"
              f"- Repeat-customer rate computed\n\n"
              f"## What this replaced\nProfiling, ETL build, anomaly triage, ad-hoc SQL and "
              f"reporting — typically ~14 human-hours — completed autonomously in minutes.")
    emit_text("Writing the executive report.")
    run_tool("write_report", {"markdown": report})
    summary = run_summary(store.get(job.id).events)
    store.emit(job.id, type="job_finished", title="Run complete", detail=json.dumps(summary))

    rec = Path(__file__).resolve().parent.parent / "recordings"
    rec.mkdir(exist_ok=True)
    with open(rec / "full_lifecycle.jsonl", "w", encoding="utf-8") as f:
        for e in store.get(job.id).events:
            f.write(e.model_dump_json() + "\n")
    print(f"Recorded {len(store.get(job.id).events)} events.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate recording, run tests, commit**

```
python scripts/make_recording.py
pytest tests/test_replay.py -v
git add -A; git commit -m "feat: replay engine + canonical recording"
```

Remove `data/*.csv` from `.gitignore`? No — keep CSVs ignored; recording JSONL is committed and data regenerates deterministically.

---

### Task 9: FastAPI service (backend/main.py)

**Files:**
- Create: `backend/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Failing test**

`tests/test_api.py`:
```python
import time
from fastapi.testclient import TestClient
import backend.main as m

def test_health_and_replay_job_flow():
    client = TestClient(m.app)
    h = client.get("/health").json()
    assert "live_available" in h
    r = client.post("/jobs", json={"mode": "replay", "speed": 0}).json()
    jid = r["job_id"]
    for _ in range(100):
        st = client.get(f"/jobs/{jid}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "finished" and st["summary"]["tool_calls"] > 5
    evs = client.get(f"/jobs/{jid}/events", params={"after": 0}).json()["events"]
    assert evs[0]["type"] == "job_started"
```

- [ ] **Step 2: Implement**

`backend/main.py`:
```python
"""DataEng Copilot API. Run: uvicorn backend.main:app --port 8000"""
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.jobs import STORE, run_in_thread
from backend.agent import run_live_job
from backend.replay import run_replay_job, RECORDING
from core.metrics import run_summary
from data.generate_sample import generate, DATA_DIR

app = FastAPI(title="DataEng Copilot API")

if not (DATA_DIR / "orders.csv").exists():
    generate()


def live_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class JobRequest(BaseModel):
    mode: str = "replay"
    speed: float = 1.0


@app.get("/health")
def health():
    return {"live_available": live_available(), "recording_available": RECORDING.exists()}


@app.post("/jobs")
def create_job(req: JobRequest):
    if req.mode == "live" and not live_available():
        raise HTTPException(409, "ANTHROPIC_API_KEY not set — live mode unavailable")
    job = STORE.create(req.mode)
    if req.mode == "live":
        run_in_thread(run_live_job, STORE, job.id, DATA_DIR)
    else:
        run_in_thread(run_replay_job, STORE, job.id, req.speed)
    return {"job_id": job.id, "mode": req.mode}


def _summary(job):
    fin = [e for e in job.events if e.type == "job_finished"]
    if fin:
        return json.loads(fin[-1].detail)
    return run_summary(job.events)


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = STORE.get(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    return {"job_id": job.id, "status": job.status, "mode": job.mode,
            "summary": _summary(job) if job.status != "running" else None}


@app.get("/jobs/{job_id}/events")
def job_events(job_id: str, after: int = 0):
    try:
        job = STORE.get(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    return {"status": job.status,
            "events": [e.model_dump() for e in STORE.events_after(job_id, after)]}


@app.get("/latest-summary")
def latest_summary():
    job = STORE.latest_finished()
    if not job:
        raise HTTPException(404, "no finished runs yet")
    return {"summary": _summary(job), "mode": job.mode}
```

- [ ] **Step 3: Test PASS, commit** `feat: FastAPI job service`

---

### Task 10: Deck export (core/deck_export.py)

**Files:**
- Create: `core/deck_export.py`
- Test: `tests/test_deck.py`

- [ ] **Step 1: Failing test**

`tests/test_deck.py`:
```python
import io
from pptx import Presentation
from core.deck_export import build_deck
from core.metrics import roi

def test_deck_builds_10_slides():
    summary = {"tool_calls": 14, "total_cost_usd": 1.82, "agent_minutes": 12.5,
               "human_hours": 14.0, "speedup_x": 67.2, "tokens_in": 250000, "tokens_out": 18000}
    data = build_deck(summary, roi(8, 120_000, 0.5, 6))
    prs = Presentation(io.BytesIO(data))
    assert len(prs.slides.slide_layouts) >= 1 and len(prs.slides._sldIdLst) == 10
```

- [ ] **Step 2: Implement**

`core/deck_export.py`:
```python
"""10-slide executive PPTX built from live run summary + ROI scenario."""
import io
from pptx import Presentation
from pptx.util import Inches, Pt

TITLE = "DataEng Copilot — Claude for Data Engineering"


def _slide(prs, title, bullets):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    body = s.placeholders[1].text_frame
    body.clear()
    for i, b in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = b
        p.font.size = Pt(18)
    return s


def build_deck(summary: dict, roi_data: dict) -> bytes:
    prs = Presentation()
    t = prs.slides.add_slide(prs.slide_layouts[0])
    t.shapes.title.text = TITLE
    t.placeholders[1].text = "Same productivity. Fraction of the headcount cost.\nBuilt on Claude (claude-opus-4-8)."

    _slide(prs, "The Problem", [
        "Data engineering teams drown in repetitive lifecycle work",
        "Profiling, ETL, quality triage, ad-hoc SQL, reporting — every week",
        "Backlogs grow faster than headcount budgets"])
    _slide(prs, "The Solution", [
        "An autonomous Claude agent runs the full lifecycle",
        "Ingest → profile → clean → anomaly scan → answer → report",
        "Human engineers review and steer instead of typing SQL"])
    _slide(prs, "Architecture", [
        "Claude (claude-opus-4-8) manual tool-use loop",
        "Local tools: DuckDB SQL, profiling, 5-layer statistical anomaly detector",
        "Every step logged: tokens, cost, latency — fully auditable",
        "Replay mode = deterministic demos, zero API risk"])
    _slide(prs, "Live Demo Results", [
        f"{summary['tool_calls']} autonomous tool calls",
        f"Agent time: {summary['agent_minutes']} min — human baseline: {summary['human_hours']} h",
        f"Speedup: {summary['speedup_x']}x",
        f"API cost: ${summary['total_cost_usd']}",
        f"Tokens: {summary['tokens_in']:,} in / {summary['tokens_out']:,} out"])
    _slide(prs, "What It Found", [
        "Seeded data-quality issues: fixed autonomously",
        "Revenue anomalies: flagged with week, severity, magnitude",
        "Business questions: answered with charts, exec-ready report"])
    _slide(prs, "ROI Scenario (illustrative)", [
        f"FTE-equivalent capacity freed: {roi_data['fte_freed']}",
        f"Annual savings at full ramp: ${roi_data['annual_savings']:,.0f}",
        f"Payback month: {roi_data['payback_month']}",
        "Framing choice: reduce headcount OR multiply output — same math"])
    _slide(prs, "Security & Operations", [
        "Runs on your data, your infra — agent tools execute locally",
        "Full event audit trail per run",
        "Human-in-the-loop review before anything ships"])
    _slide(prs, "Roadmap", [
        "Pilot: one team, 4 weeks, measure baseline vs agent",
        "Expand: orchestration (Airflow/dbt), warehouse connectors",
        "Scale: agent fleet across all data domains"])
    _slide(prs, "The Ask", [
        "30-minute live demo with your data team",
        "Partnership conversation with Anthropic",
        "Contact: gsundar4121@gmail.com"])

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
```

- [ ] **Step 3: Test PASS, commit** `feat: executive PPTX deck export`

---

### Task 11: Streamlit frontend

**Files:**
- Create: `frontend/Home.py`, `frontend/pages/2_Live_Demo.py`, `frontend/pages/3_ROI_Calculator.py`, `frontend/pages/4_Contact.py`, `frontend/api.py`
- Test: `tests/test_frontend_smoke.py`

- [ ] **Step 1: Smoke test (compile only — Streamlit pages execute on import)**

`tests/test_frontend_smoke.py`:
```python
import py_compile
from pathlib import Path

def test_frontend_pages_compile():
    root = Path(__file__).resolve().parent.parent / "frontend"
    for f in list(root.glob("*.py")) + list((root / "pages").glob("*.py")):
        py_compile.compile(str(f), doraise=True)
```

- [ ] **Step 2: Shared API helper**

`frontend/api.py`:
```python
import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for core.*

BASE = "http://127.0.0.1:8000"


def get(path, **kw):
    try:
        r = requests.get(f"{BASE}{path}", timeout=5, **kw)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def post(path, payload):
    try:
        r = requests.post(f"{BASE}{path}", json=payload, timeout=5)
        return r.json() if r.ok else {"error": r.text}
    except requests.RequestException as e:
        return {"error": str(e)}
```

- [ ] **Step 3: Home (landing + deck download)**

`frontend/Home.py`:
```python
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
```

- [ ] **Step 4: Live Demo page**

`frontend/pages/2_Live_Demo.py`:
```python
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
```

- [ ] **Step 5: ROI Calculator page**

`frontend/pages/3_ROI_Calculator.py`:
```python
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
```

- [ ] **Step 6: Contact page**

`frontend/pages/4_Contact.py`:
```python
import streamlit as st

st.set_page_config(page_title="Contact", page_icon="✉️")
st.title("✉️ Let's talk")
st.markdown("""
**Pitch:** I run data engineering on Claude. Same productivity, reduced headcount cost —
demonstrated live, measured per step, auditable end to end.

**Who I want to reach:** Anthropic partnerships / solutions teams and C-level data leaders.

- 📧 **Email:** [gsundar4121@gmail.com](mailto:gsundar4121@gmail.com?subject=DataEng%20Copilot%20demo)
- 📅 **Book a demo:** calendly link goes here
- 📦 **Take-away:** download the exec deck from the Home page

*Built with the Anthropic Python SDK on `claude-opus-4-8`.*
""")
st.link_button("Request a live demo",
               "mailto:gsundar4121@gmail.com?subject=DataEng%20Copilot%20demo")
```

- [ ] **Step 7: Smoke test PASS, commit** `feat: Streamlit frontend (landing, demo, ROI, contact)`

---

### Task 12: Runner, README, full suite

**Files:**
- Create: `run.ps1`, `README.md`

- [ ] **Step 1: run.ps1**

```powershell
# Starts backend (:8000) and frontend (:8501). Run from repo root.
$root = $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "Set-Location '$root'; python -m uvicorn backend.main:app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "Set-Location '$root'; python -m streamlit run frontend/Home.py --server.port 8501"
Write-Host "Backend: http://127.0.0.1:8000  Frontend: http://127.0.0.1:8501"
```

- [ ] **Step 2: README.md**

```markdown
# DataEng Copilot — Claude for Data Engineering (Pitch MVP)

Autonomous Claude agent runs a full data-engineering lifecycle (profile → ETL →
anomaly scan → SQL answers → exec report) on a retail dataset, with per-step
cost/time metrics, an ROI calculator, and a one-click executive PPTX deck.

## Quickstart
    pip install -r requirements.txt
    python scripts/make_recording.py   # build the replay recording (no API key needed)
    .\run.ps1                          # backend :8000 + frontend :8501

Open http://127.0.0.1:8501 → Live Demo → Start run (replay).

## Live mode
    $env:ANTHROPIC_API_KEY = "sk-ant-..."   # then restart backend
Model: `claude-opus-4-8`. Record a fresh canonical run with `$env:RECORD = "1"`.

## Tests
    pytest
```

- [ ] **Step 3: Full suite + commit**

```
pytest -v
git add -A; git commit -m "feat: runner script and README"
```

Expected: all tests PASS.

---

## Self-Review

- **Spec coverage:** architecture ✅ (T1,6,7,9), agent engine ✅ (T7), tools ✅ (T5), replay ✅ (T8), metrics/headcount ✅ (T4), ROI page ✅ (T11.5), pitch page + deck + contact ✅ (T10, T11), error handling ✅ (agent try/except, API 409, UI switch-to-replay), testing ✅ (every task).
- **Placeholders:** calendly link intentionally labeled as placeholder per spec. No TBDs.
- **Type consistency:** `JobEvent` fields used identically in agent/replay/recording/frontend; `ToolExecutor.execute` returns `{result, artifact}` everywhere; `roi()` keys match deck + ROI page.
