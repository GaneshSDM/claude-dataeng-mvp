"""Adapter: DuckDB retail orders -> vendor MultiLayerAnomalyDetector facts."""
import pandas as pd
from backend.vendor_detector import DetectorConfig, MultiLayerAnomalyDetector


def scan_orders(conn) -> list[dict]:
    weekly = conn.execute("""
        SELECT date_trunc('week', CAST(order_date AS DATE)) AS week_start_date,
               channel,
               SUM(quantity * TRY_CAST(unit_price AS DOUBLE)) AS gdv,
               COUNT(*) AS completes
        FROM orders
        WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1
    """).fetchdf()
    facts = pd.DataFrame({
        "SOURCE_TABLE": "ORDERS",
        "WEEK_START_DATE": pd.to_datetime(weekly["week_start_date"]),
        "COUNTRY": "ALL",
        "DEVICE": weekly["channel"],
        "TRAFFIC_SOURCE": "ALL",
        "GDV": weekly["gdv"],
        "COMPLETES": weekly["completes"],
    })
    det = MultiLayerAnomalyDetector(DetectorConfig())
    res = det.detect(facts)
    hits = res[res["direction"] != "NORMAL"].sort_values(
        "ensemble_score", key=abs, ascending=False)
    return [{
        "week": str(r.week_start_date)[:10], "metric": r.metric, "direction": r.direction,
        "severity": r.severity, "pct_diff": round(float(r.pct_diff), 1),
        "z_score": round(float(r.z_score), 2), "actual": round(float(r.actual), 2),
        "baseline_avg": round(float(r.baseline_avg), 2),
    } for r in hits.head(10).itertuples()]
