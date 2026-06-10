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
