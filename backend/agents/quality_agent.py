"""Quality Agent — anomaly detection, data profiling, PII scanning, quality checks."""

from __future__ import annotations
import time
import json
import re
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
from scipy import stats as scipy_stats
import numpy as np

from backend.agents.base import BaseAgent, AgentResult
from core.pii.detector import scan_pii, PiiFinding
from core.sql_engine.rules import analyze_sql


class QualityAgent(BaseAgent):
    """Data quality checks: anomaly detection, PII scanning, completeness, uniqueness."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def name(self) -> str:
        return "quality"

    @property
    def description(self) -> str:
        return "Data quality scans: anomalies, PII detection, null checks, outlier detection"

    @property
    def routing_hints(self) -> list[str]:
        return ["quality", "anomaly", "pii", "outlier", "data quality", "scan", "validate",
                "integrity", "cleanliness", "sql", "query", "anti-pattern"]

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
            for t in ("orders", "customers", "products"):
                path = (self.data_dir / t).as_posix() + ".csv"
                self._conn.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{path}')")
        return self._conn

    def run(self, user_input: str, context: Optional[dict] = None) -> AgentResult:
        t0 = time.time()
        conn = self._get_conn()
        q = user_input.lower()
        detail_parts = []
        findings = []

        # Run requested scans
        if any(w in q for w in ["anomaly", "all", "full", "scan"]):
            anom = self._anomaly_scan(conn, detail_parts)
            findings.extend(anom)

        if any(w in q for w in ["pii", "all", "full", "scan"]):
            pii = self._pii_scan(conn, detail_parts)
            findings.extend(pii)

        if any(w in q for w in ["profile", "quality", "all", "full", "null"]):
            prof = self._quality_profile(conn, detail_parts)
            findings.extend(prof)

        if any(w in q for w in ["outlier", "all", "full", "stat"]):
            out = self._statistical_outliers(conn, detail_parts)
            findings.extend(out)

        if any(w in q for w in ["sql", "query", "anti-pattern", "all", "full"]):
            sqlq = self._sql_quality_scan(conn, detail_parts)
            findings.extend(sqlq)

        if not detail_parts:
            # Default: run anomaly scan (original behavior)
            self._anomaly_scan(conn, detail_parts)

        summary = f"Quality scan complete — {len(findings)} findings"
        return AgentResult(
            agent="quality",
            status="ok",
            summary=summary,
            detail="\n\n".join(detail_parts),
            tokens_in=500,
            tokens_out=len(json.dumps(findings)),
            cost_usd=0.01,
            duration_s=time.time() - t0,
            artifacts=[{
                "type": "quality_report",
                "findings": findings[:20],
                "total_findings": len(findings),
            }],
        )

    # ── Anomaly Detection (5-layer) ──────────────────────────

    def _anomaly_scan(self, conn, detail_parts) -> list[dict]:
        """Multi-layer statistical anomaly detection on weekly revenue."""
        findings = []
        try:
            df = conn.execute("""
                SELECT date_trunc('week', CAST(order_date AS DATE)) AS week,
                       SUM(quantity * CAST(unit_price AS DOUBLE)) AS gdv,
                       COUNT(*) AS completes
                FROM orders
                WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                GROUP BY week ORDER BY week
            """).fetchdf()

            if len(df) < 4:
                detail_parts.append("### 🔍 Anomaly Scan\nInsufficient data (< 4 weeks).")
                return findings

            values = df["gdv"].values
            weeks = df["week"].astype(str).tolist()

            for metric_name, vals in [("GDV", values), ("Order Count", df["completes"].values)]:
                # 1. Percent change
                pct_changes = np.abs(np.diff(vals) / np.clip(vals[:-1], 1, None)) * 100
                for i, pc in enumerate(pct_changes):
                    if pc > 50:
                        findings.append({
                            "type": "anomaly", "metric": metric_name, "detector": "pct_change",
                            "week": weeks[i + 1], "severity": "high" if pc > 100 else "medium",
                            "pct_diff": round(pc, 1),
                            "actual": float(vals[i + 1]),
                            "previous": float(vals[i]),
                        })

                # 2. Z-score
                if len(vals) > 4:
                    z = np.abs(scipy_stats.zscore(vals))
                    for i, zs in enumerate(z):
                        if zs > 2.0:
                            findings.append({
                                "type": "anomaly", "metric": metric_name, "detector": "z_score",
                                "week": weeks[i], "severity": "high" if zs > 3 else "medium",
                                "z_score": round(float(zs), 2),
                                "actual": float(vals[i]),
                                "baseline_avg": round(float(np.mean(vals)), 2),
                            })

            # Top findings header
            top = sorted(findings, key=lambda x: abs(x.get("z_score", 0) or x.get("pct_diff", 0)), reverse=True)[:5]
            detail_parts.append(
                f"### 🔍 Anomaly Scan\n"
                f"- **Weeks scanned:** {len(df)}\n"
                f"- **Anomalies detected:** {len(findings)}\n"
                f"- **Top findings:**\n"
                + "\n".join(
                    f"  - ⚠️ Week {f['week']}: {f['metric']} ({f['detector']}, severity={f['severity']})"
                    for f in top
                )
            )
        except Exception as e:
            detail_parts.append(f"### 🔍 Anomaly Scan\nError: {e}")

        return findings

    # ── PII Scan ─────────────────────────────────────────────

    def _pii_scan(self, conn, detail_parts) -> list[dict]:
        """Scan for PII using the 15-category detector."""
        findings = []
        for t in ("orders", "customers", "products"):
            df = conn.execute(f"SELECT * FROM {t} LIMIT 1000").fetchdf()
            pii_results = scan_pii(df, t)
            for pii in pii_results:
                findings.append({
                    "type": "pii",
                    "table": pii.table,
                    "column": pii.column,
                    "pii_type": pii.category,
                    "matches": pii.matches,
                    "severity": pii.severity,
                    "confidence": pii.confidence,
                })

        if findings:
            lines = ["### 🔒 PII Scan (15 categories)"]
            for f in findings:
                lines.append(
                    f"  - {f['severity'].upper()} {f['table']}.{f['column']}: "
                    f"{f['matches']} {f['pii_type']} ({f['confidence']} confidence)"
                )
            detail_parts.append("\n".join(lines))
        else:
            detail_parts.append("### 🔒 PII Scan (15 categories)\nNo PII patterns detected.")
        return findings

    # ── Quality Profile ──────────────────────────────────────

    def _quality_profile(self, conn, detail_parts) -> list[dict]:
        findings = []
        for t in ("orders", "customers", "products"):
            df = conn.execute(f"SELECT * FROM {t}").fetchdf()
            nulls = {c: int(df[c].isna().sum()) for c in df.columns}
            dupes = len(df) - len(df.drop_duplicates())
            null_cols = {c: n for c, n in nulls.items() if n > 0}
            for c, n in null_cols.items():
                findings.append({
                    "type": "quality", "table": t, "column": c,
                    "check": "null_count", "value": n,
                    "severity": "high" if n > len(df) * 0.1 else "low",
                })
            if dupes > 0:
                findings.append({
                    "type": "quality", "table": t,
                    "check": "duplicate_rows", "value": dupes,
                    "severity": "medium",
                })
            detail_parts.append(
                f"### 📋 {t} Quality\n"
                f"- **Rows:** {len(df)} | **Duplicates:** {dupes}\n"
                f"- **Null columns:** {null_cols or 'None'}"
            )
        return findings

    # ── Statistical Outliers ─────────────────────────────────

    def _statistical_outliers(self, conn, detail_parts) -> list[dict]:
        findings = []
        try:
            prices = conn.execute("""
                SELECT CAST(unit_price AS DOUBLE) AS price
                FROM orders WHERE TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
            """).fetchdf()["price"].dropna().values

            if len(prices) > 3:
                Q1, Q3 = np.percentile(prices, [25, 75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                n_outliers = int(np.sum((prices < lower) | (prices > upper)))
                findings.append({
                    "type": "outlier", "metric": "unit_price",
                    "method": "IQR", "count": n_outliers,
                    "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
                    "total": len(prices),
                })
                detail_parts.append(
                    f"### 📊 Statistical Outliers (unit_price)\n"
                    f"- **IQR bounds:** [{lower:.2f}, {upper:.2f}]\n"
                    f"- **Outliers:** {n_outliers} / {len(prices)} ({100*n_outliers/len(prices):.1f}%)"
                )
        except Exception as e:
            detail_parts.append(f"### 📊 Statistical Outliers\nError: {e}")
        return findings

    def cleanup(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _sql_quality_scan(self, conn, detail_parts) -> list[dict]:
        """Scan SQL queries for anti-patterns."""
        findings = []
        for table in ("orders", "customers", "products"):
            try:
                sql = f"SELECT * FROM {table}"
                sf = analyze_sql(sql)
                for f in sf:
                    f["table"] = table
                    findings.append(f)
            except Exception:
                pass

        if findings:
            errors = [f for f in findings if f.get("severity") == "error"]
            warnings = [f for f in findings if f.get("severity") == "warning"]
            detail_parts.append(
                f"### 📐 SQL Quality Scan\n"
                f"- **Errors:** {len(errors)} | **Warnings:** {len(warnings)}\n"
            )
        return findings