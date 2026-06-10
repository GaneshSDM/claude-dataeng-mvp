"""Build recordings/full_lifecycle.jsonl by running real tools with scripted narration.
Usage: python scripts/make_recording.py   (or RECORD=1 live run via backend/agent.py)

Tool outputs are genuine (real DuckDB + real anomaly detector); only the narration text
and token numbers are scripted. The UI labels replays honestly as replay mode."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generate_sample import generate, DATA_DIR  # noqa: E402
from backend.jobs import JobStore  # noqa: E402
from backend.tools import ToolExecutor, step_for  # noqa: E402
from core.metrics import MODEL_ID, cost_usd, run_summary  # noqa: E402

SCRIPT = [
    ("Profiling the three raw tables first.", "profile_data", {"table": "orders"}),
    (None, "profile_data", {"table": "customers"}),
    (None, "profile_data", {"table": "products"}),
    ("Raw data has negative quantities, non-numeric prices and duplicate customers — building clean tables.",
     "run_sql", {"purpose": "etl", "query":
        "CREATE TABLE orders_clean AS SELECT order_id, customer_id, product_id, "
        "CAST(order_date AS DATE) AS order_date, quantity, "
        "TRY_CAST(unit_price AS DOUBLE) AS unit_price, channel, "
        "quantity * TRY_CAST(unit_price AS DOUBLE) AS revenue "
        "FROM orders WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL"}),
    (None, "run_sql", {"purpose": "etl", "query":
        "CREATE TABLE customers_clean AS SELECT DISTINCT * FROM customers"}),
    (None, "run_sql", {"purpose": "etl", "query":
        "CREATE TABLE weekly_sales AS SELECT date_trunc('week', order_date) AS week, "
        "SUM(revenue) AS revenue, COUNT(*) AS order_count FROM orders_clean GROUP BY 1 ORDER BY 1"}),
    ("Marts ready. Running the statistical anomaly scan.", "run_anomaly_scan", {}),
    ("Answering the business questions.", "run_sql", {"purpose": "question", "query":
        "SELECT p.product_name, ROUND(SUM(o.revenue),2) AS revenue FROM orders_clean o "
        "JOIN products p USING(product_id) GROUP BY 1 ORDER BY 2 DESC LIMIT 5"}),
    (None, "run_sql", {"purpose": "question", "query":
        "SELECT date_trunc('month', order_date) AS month, ROUND(SUM(revenue),2) AS revenue "
        "FROM orders_clean GROUP BY 1 ORDER BY 1"}),
    (None, "run_sql", {"purpose": "question", "query":
        "SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE n > 1) / COUNT(*), 1) AS repeat_pct "
        "FROM (SELECT customer_id, COUNT(*) AS n FROM orders_clean GROUP BY 1)"}),
]


def main():
    if not (DATA_DIR / "orders.csv").exists():
        generate()
    store = JobStore()
    job = store.create("live")  # canonical run; replay engine re-flags mode itself
    ex = ToolExecutor(DATA_DIR)
    store.emit(job.id, type="job_started", title="Full lifecycle demo", detail=MODEL_ID)

    def emit_text(text, ti=2400, to=120):
        store.emit(job.id, type="claude_text", detail=text, tokens_in=ti, tokens_out=to,
                   cost_usd=cost_usd(ti, to), duration_s=2.2)
        time.sleep(0.05)

    def run_tool(name, args):
        step = step_for(name, args)
        store.emit(job.id, type="tool_call", step=step, title=name,
                   detail=json.dumps(args)[:500],
                   sql=args.get("query") if name == "run_sql" else None)
        t0 = time.time()
        out = ex.execute(name, args)
        ev = ("chart" if (out["artifact"] or {}).get("type") == "chart" else
              "report" if (out["artifact"] or {}).get("type") == "report" else "tool_result")
        store.emit(job.id, type=ev, step=step, title=name,
                   detail=json.dumps(out["result"], default=str)[:1500],
                   artifact=out["artifact"], duration_s=max(time.time() - t0, 0.4))
        return out

    monthly = None
    top5 = None
    findings = None
    for narration, name, args in SCRIPT:
        if narration:
            emit_text(narration)
        out = run_tool(name, args)
        if name == "run_anomaly_scan":
            findings = out["result"]["findings"]
        if name == "run_sql" and "product_name" in args.get("query", ""):
            top5 = out["result"]
        if name == "run_sql" and "month" in args.get("query", ""):
            monthly = out["result"]

    emit_text("Building charts for the executive report.")
    run_tool("create_chart", {"title": "Monthly Revenue Trend", "chart_type": "line",
                              "x": [str(r[0])[:10] for r in monthly["rows"]],
                              "y": [float(r[1]) for r in monthly["rows"]],
                              "x_label": "Month", "y_label": "Revenue (USD)"})
    run_tool("create_chart", {"title": "Top 5 Products by Revenue", "chart_type": "bar",
                              "x": [str(r[0]) for r in top5["rows"]],
                              "y": [float(r[1]) for r in top5["rows"]],
                              "x_label": "Product", "y_label": "Revenue (USD)"})
    flines = "\n".join(f"- {f['week']} **{f['metric']} {f['direction']}** ({f['severity']}), "
                       f"{f['pct_diff']}% vs baseline" for f in (findings or [])[:5])
    report = (f"# Executive Summary — Retail Data Lifecycle\n\n"
              f"## Data quality fixes\n- Removed negative-quantity orders\n"
              f"- Dropped non-numeric prices\n- Deduplicated customers\n\n"
              f"## Anomaly findings (5-layer detector)\n{flines}\n\n"
              f"## Business answers\n- Top products and monthly trend charted\n"
              f"- Repeat-customer rate computed\n\n"
              f"## What this replaced\nProfiling, ETL build, anomaly triage, ad-hoc SQL and "
              f"reporting — typically ~14 human-hours — completed autonomously in minutes.")
    emit_text("Writing the executive report.")
    run_tool("write_report", {"markdown": report})
    summary = run_summary(store.get(job.id).events)
    store.emit(job.id, type="job_finished", title="Run complete", detail=json.dumps(summary))

    rec = Path(__file__).resolve().parent.parent / "recordings"
    rec.mkdir(exist_ok=True)
    with open(rec / "full_lifecycle.jsonl", "w", encoding="utf-8") as f:
        for e in store.get(job.id).events:
            f.write(e.model_dump_json() + "\n")
    print(f"Recorded {len(store.get(job.id).events)} events.")


if __name__ == "__main__":
    main()
