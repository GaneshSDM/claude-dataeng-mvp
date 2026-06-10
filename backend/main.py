"""DataEng Copilot API — Multi-Agent Orchestration. Run: uvicorn backend.main:app --port 8000"""
from __future__ import annotations
import json
import os
import time
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.jobs import STORE, run_in_thread
from backend.agent import run_live_job
from backend.replay import run_replay_job, RECORDING
from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.base import AgentResult
from core.metrics import run_summary
from data.generate_sample import generate, DATA_DIR

app = FastAPI(title="DataEng Copilot API — Multi-Agent")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if not (DATA_DIR / "orders.csv").exists():
    generate()

# ── Agent registry ──────────────────────────────────────────
_orchestrator: Optional[OrchestratorAgent] = None
_orch_lock = threading.Lock()


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        with _orch_lock:
            if _orchestrator is None:
                _orchestrator = OrchestratorAgent(DATA_DIR)
    return _orchestrator


def live_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Schemas ─────────────────────────────────────────────────

class JobRequest(BaseModel):
    mode: str = "replay"
    speed: float = 1.0


class AgentRequest(BaseModel):
    agent: str = ""                     # "" = auto-route via orchestrator
    input: str
    context: Optional[dict] = None


class PipelineStep(BaseModel):
    agent: str
    input: str


class PipelineRequest(BaseModel):
    steps: list[PipelineStep]


# ── Health & Info ───────────────────────────────────────────

@app.get("/health")
def health():
    agents = get_orchestrator().available_agents
    return {
        "live_available": live_available(),
        "recording_available": RECORDING.exists(),
        "agents": agents,
        "agent_count": len(agents),
    }


# ── Original job endpoints (keep backward compat) ───────────

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


@app.get("/latest-summary")
def latest_summary():
    job = STORE.latest_finished()
    if not job:
        raise HTTPException(404, "no finished runs yet")
    return {"summary": _summary(job), "mode": job.mode}


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


# ── NEW: Agent endpoints ────────────────────────────────────

@app.get("/agents")
def list_agents():
    """List all available agents and their capabilities."""
    orch = get_orchestrator()
    return {"agents": orch.available_agents}


@app.post("/agents/run")
def run_agent(req: AgentRequest):
    """Run a single agent or auto-route via the orchestrator."""
    t0 = time.time()
    orch = get_orchestrator()

    if req.agent and req.agent != "orchestrator":
        agent = orch.get_agent(req.agent.lower())
        if not agent:
            raise HTTPException(404, f"Unknown agent '{req.agent}'. Available: {[a['name'] for a in orch.available_agents]}")
        result = agent.run(req.input, req.context)
    else:
        result = orch.run(req.input, req.context)

    result.duration_s = time.time() - t0
    return _result_to_response(result)


@app.post("/agents/pipeline")
def run_pipeline(req: PipelineRequest):
    """Run a multi-step pipeline (each step is an agent invocation)."""
    t0 = time.time()
    orch = get_orchestrator()
    results = []

    for i, step in enumerate(req.steps):
        agent = orch.get_agent(step.agent.lower())
        if not agent:
            results.append({
                "step": i, "agent": step.agent, "status": "error",
                "error": f"Unknown agent '{step.agent}'",
            })
            continue
        try:
            result = agent.run(step.input)
            results.append({
                "step": i, "agent": step.agent, "status": result.status,
                "summary": result.summary, "detail": result.detail[:1000],
                "duration_s": round(result.duration_s, 2),
                "artifacts": result.artifacts,
                "sql": result.sql,
            })
        except Exception as e:
            results.append({
                "step": i, "agent": step.agent, "status": "error",
                "error": str(e),
            })

    return {
        "pipeline_status": "ok" if all(r["status"] == "ok" for r in results) else "partial",
        "steps": len(req.steps),
        "total_duration_s": round(time.time() - t0, 2),
        "results": results,
    }


# ── Helpers ─────────────────────────────────────────────────

def _result_to_response(r: AgentResult) -> dict:
    return {
        "agent": r.agent,
        "status": r.status,
        "summary": r.summary,
        "detail": r.detail,
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "cost_usd": round(r.cost_usd, 6),
        "duration_s": round(r.duration_s, 2),
        "artifacts": r.artifacts,
        "sql": r.sql,
        "error": r.error,
    }