import time
from fastapi.testclient import TestClient
import backend.main as m


def test_health_and_replay_job_flow():
    client = TestClient(m.app)
    h = client.get("/health").json()
    assert "live_available" in h and h["recording_available"]
    r = client.post("/jobs", json={"mode": "replay", "speed": 0}).json()
    jid = r["job_id"]
    st = {}
    for _ in range(100):
        st = client.get(f"/jobs/{jid}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "finished" and st["summary"]["tool_calls"] > 5
    evs = client.get(f"/jobs/{jid}/events", params={"after": 0}).json()["events"]
    assert evs[0]["type"] == "job_started"
    assert client.get("/latest-summary").json()["summary"]["tool_calls"] > 5
