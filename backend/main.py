"""DataEng Copilot API. Run: uvicorn backend.main:app --port 8000"""
import json
import os
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
