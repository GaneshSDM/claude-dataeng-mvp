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
