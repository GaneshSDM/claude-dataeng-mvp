from core.events import JobEvent


def test_job_event_defaults_and_roundtrip():
    e = JobEvent(seq=1, ts=123.4, type="tool_call", step="etl", title="run_sql")
    j = e.model_dump_json()
    e2 = JobEvent.model_validate_json(j)
    assert e2 == e and e2.cost_usd == 0.0 and e2.artifact is None
