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
