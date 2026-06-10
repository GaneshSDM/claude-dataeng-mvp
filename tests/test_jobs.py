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
