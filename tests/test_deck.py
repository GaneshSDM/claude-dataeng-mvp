import io
from pptx import Presentation
from core.deck_export import build_deck
from core.metrics import roi


def test_deck_builds_10_slides():
    summary = {"tool_calls": 14, "total_cost_usd": 1.82, "agent_minutes": 12.5,
               "human_hours": 14.0, "speedup_x": 67.2, "tokens_in": 250000, "tokens_out": 18000}
    data = build_deck(summary, roi(8, 120_000, 0.5, 6))
    prs = Presentation(io.BytesIO(data))
    assert len(prs.slides._sldIdLst) == 10
