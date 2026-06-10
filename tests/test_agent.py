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
