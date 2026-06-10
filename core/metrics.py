"""Illustrative productivity + ROI math. All baselines are assumptions, surfaced in UI."""
from core.events import JobEvent

MODEL_ID = "claude-opus-4-8"
PRICE_IN_PER_TOK = 5.0 / 1_000_000    # USD, claude-opus-4-8
PRICE_OUT_PER_TOK = 25.0 / 1_000_000

# Illustrative human-baseline hours per task type (industry-typical, adjustable story)
HUMAN_BASELINE_HOURS = {"profile": 2.0, "etl": 6.0, "anomaly": 3.0, "sql": 1.0, "report": 2.0}

PROGRAM_COST_YEAR1 = 150_000.0   # illustrative: setup + API + enablement
PROGRAM_COST_ONGOING = 50_000.0  # per year thereafter


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * PRICE_IN_PER_TOK + tokens_out * PRICE_OUT_PER_TOK


def run_summary(events: list[JobEvent]) -> dict:
    steps_done = {e.step for e in events if e.type == "tool_call" and e.step}
    human_hours = sum(HUMAN_BASELINE_HOURS.get(s, 0.0) for s in steps_done)
    agent_seconds = sum(e.duration_s for e in events)
    agent_minutes = round(agent_seconds / 60, 2)
    total_cost = round(sum(e.cost_usd for e in events), 4)
    speedup = round((human_hours * 60) / agent_minutes, 1) if agent_minutes else 0.0
    return {
        "steps_done": sorted(steps_done),
        "tool_calls": sum(1 for e in events if e.type == "tool_call"),
        "tokens_in": sum(e.tokens_in for e in events),
        "tokens_out": sum(e.tokens_out for e in events),
        "total_cost_usd": total_cost,
        "agent_minutes": agent_minutes,
        "human_hours": human_hours,
        "speedup_x": speedup,
    }


def roi(team_size: int, loaded_cost: float, automatable_pct: float, ramp_months: int) -> dict:
    """24-month illustrative model. automatable_pct of task-hours absorbed at full ramp."""
    fte_freed = round(team_size * automatable_pct, 1)
    annual_savings = fte_freed * loaded_cost
    monthly_net, cum = [], 0.0
    payback_month = None
    for m in range(1, 25):
        ramp = min(1.0, m / max(ramp_months, 1))
        savings = (annual_savings / 12) * ramp
        cost = (PROGRAM_COST_YEAR1 / 12) if m <= 12 else (PROGRAM_COST_ONGOING / 12)
        cum += savings - cost
        monthly_net.append(round(cum, 0))
        if payback_month is None and cum > 0:
            payback_month = m
    return {"fte_freed": fte_freed, "annual_savings": annual_savings,
            "monthly_net": monthly_net, "payback_month": payback_month}
