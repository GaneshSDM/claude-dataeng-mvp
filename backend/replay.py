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
