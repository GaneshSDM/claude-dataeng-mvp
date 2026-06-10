"""Shared job event schema — identical for live and replay."""
from typing import Literal, Optional, Any
from pydantic import BaseModel

EventType = Literal["job_started", "step_started", "claude_text", "tool_call",
                    "tool_result", "chart", "report", "job_finished", "job_failed"]


class JobEvent(BaseModel):
    seq: int
    ts: float
    type: EventType
    step: str = ""            # profile|etl|anomaly|sql|report|"" — keys into metrics baselines
    title: str = ""
    detail: str = ""
    sql: Optional[str] = None
    artifact: Optional[dict[str, Any]] = None  # inline chart/report payload
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    mode: Literal["live", "replay"] = "live"
