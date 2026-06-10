"""Manual Claude tool-use loop. Manual (not tool_runner) for per-step events,
token accounting, and recording."""
import json
import os
import time
from pathlib import Path
from backend.jobs import JobStore
from backend.tools import TOOL_DEFS, ToolExecutor, step_for
from core.metrics import MODEL_ID, cost_usd, run_summary

SYSTEM = """You are DataEng Copilot, an autonomous data engineering agent demoing a full
lifecycle on a retail dataset (DuckDB tables: orders, customers, products — raw, dirty).
Mission, in order:
1. profile_data each raw table.
2. ETL with run_sql(purpose=etl): build orders_clean (drop negative quantities, cast
   unit_price dropping non-numeric, dedupe customers into customers_clean) and a
   weekly_sales mart (week, revenue, order_count).
3. run_anomaly_scan once.
4. Answer 3 business questions with run_sql(purpose=question): top 5 products by revenue;
   monthly revenue trend; repeat-customer rate.
5. create_chart twice: monthly revenue trend (line), top products (bar).
6. write_report once: executive markdown summary — data quality fixes, anomaly findings
   with weeks and severity, business answers, and a 'what this replaced' note.
Be decisive; no questions. Keep narration to one short sentence before each action."""

KICKOFF = "Run the full lifecycle now."


def run_live_job(store: JobStore, job_id: str, data_dir: Path, client=None, max_iters: int = 30):
    record = os.environ.get("RECORD") == "1"
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        ex = ToolExecutor(data_dir)
        store.emit(job_id, type="job_started", title="Full lifecycle demo", detail=MODEL_ID)
        messages = [{"role": "user", "content": KICKOFF}]
        for _ in range(max_iters):
            t0 = time.time()
            resp = client.messages.create(
                model=MODEL_ID, max_tokens=16000, system=SYSTEM,
                thinking={"type": "adaptive"}, output_config={"effort": "high"},
                tools=TOOL_DEFS, messages=messages)
            dt = time.time() - t0
            c = cost_usd(resp.usage.input_tokens, resp.usage.output_tokens)
            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            first = True
            for b in resp.content:
                if getattr(b, "type", "") == "text" and b.text.strip():
                    store.emit(job_id, type="claude_text", detail=b.text,
                               tokens_in=resp.usage.input_tokens if first else 0,
                               tokens_out=resp.usage.output_tokens if first else 0,
                               cost_usd=c if first else 0.0,
                               duration_s=dt if first else 0.0)
                    first = False
            if first and tool_uses:
                # no text block carried the usage — attach it to the first tool_call below
                pass
            if resp.stop_reason != "tool_use" or not tool_uses:
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                step = step_for(tu.name, tu.input)
                store.emit(job_id, type="tool_call", step=step, title=tu.name,
                           detail=json.dumps(tu.input)[:500],
                           sql=tu.input.get("query") if tu.name == "run_sql" else None,
                           tokens_in=resp.usage.input_tokens if first else 0,
                           tokens_out=resp.usage.output_tokens if first else 0,
                           cost_usd=c if first else 0.0,
                           duration_s=dt if first else 0.0)
                first = False
                t1 = time.time()
                try:
                    out = ex.execute(tu.name, tu.input)
                    payload, is_err = json.dumps(out["result"], default=str)[:4000], False
                except Exception as e:  # tool failure -> let Claude adapt
                    out, payload, is_err = {"artifact": None}, f"Error: {e}", True
                ev_type = ("chart" if (out["artifact"] or {}).get("type") == "chart"
                           else "report" if (out["artifact"] or {}).get("type") == "report"
                           else "tool_result")
                store.emit(job_id, type=ev_type, step=step, title=tu.name,
                           detail=payload[:1500], artifact=out["artifact"],
                           duration_s=time.time() - t1)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": payload, "is_error": is_err})
            messages.append({"role": "user", "content": results})
        summary = run_summary(store.get(job_id).events)
        store.emit(job_id, type="job_finished", title="Run complete",
                   detail=json.dumps(summary))
        store.finish(job_id)
        if record:
            _dump_recording(store, job_id)
    except Exception as e:
        store.emit(job_id, type="job_failed", title="Job failed", detail=str(e)[:800])
        store.finish(job_id, failed=True)


def _dump_recording(store: JobStore, job_id: str):
    rec_dir = Path(__file__).resolve().parent.parent / "recordings"
    rec_dir.mkdir(exist_ok=True)
    with open(rec_dir / "full_lifecycle.jsonl", "w", encoding="utf-8") as f:
        for e in store.get(job_id).events:
            f.write(e.model_dump_json() + "\n")
