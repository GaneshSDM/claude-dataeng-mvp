"""Analytics Agent — generates reports, summaries, charts, and insights from data."""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
import numpy as np

from backend.agents.base import BaseAgent, AgentResult


class AnalyticsAgent(BaseAgent):
    """Generates structured reports, visualizations, and business insights from data."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def name(self) -> str:
        return "analytics"

    @property
    def description(self) -> str:
        return "Generate reports, charts, and business insights from your data"

    @property
    def routing_hints(self) -> list[str]:
        return ["report", "chart", "visualize", "dashboard", "insight", "summary",
                "executive", "kpi", "metric", "analytics", "business"]

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
        artifacts = []

        # If user asks for a specific report type, generate that
        if "roi" in q:
            report = self._roi_report(conn)
        elif "executive" in q or "summary" in q:
            report = self._executive_summary(conn)
        elif "kpi" in q or "metric" in q or "key" in q:
            report = self._kpi_report(conn)
        elif "comparison" in q or "segment" in q:
            report = self._segmentation(conn)
        else:
            report = self._executive_summary(conn)

        # Build charts as artifacts
        charts = self._build_charts(conn, q)
        artifacts.extend(charts)

        return AgentResult(
            agent="analytics",
            status="ok",
            summary=report["summary"],
            detail=report["detail"],
            artifacts=artifacts,
            tokens_in=report.get("tokens_in", 500),
            tokens_out=report.get("tokens_out", 1000),
            cost_usd=0.02,
            duration_s=time.time() - t0,
        )

    # ── Report generators ────────────────────────────────────

    def _executive_summary(self, conn) -> dict:
        """Generate a full executive summary."""
        sections = []

        # Data quality
        total_orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        bad_qty = conn.execute("SELECT COUNT(*) FROM orders WHERE quantity <= 0 OR quantity IS NULL").fetchone()[0]
        bad_price = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE TRY_CAST(unit_price AS DOUBLE) IS NULL").fetchone()[0]

        # Revenue
        rev = conn.execute("""
            SELECT ROUND(SUM(quantity * CAST(unit_price AS DOUBLE)), 2)
            FROM orders WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
        """).fetchone()[0] or 0

        # Weekly trend
        weekly = conn.execute("""
            SELECT date_trunc('week', CAST(order_date AS DATE)) AS week,
                   ROUND(SUM(quantity * CAST(unit_price AS DOUBLE)), 2) AS revenue,
                   COUNT(*) AS orders
            FROM orders WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
            GROUP BY week ORDER BY week
        """).fetchdf()

        avg_weekly = weekly["revenue"].mean() if len(weekly) > 0 else 0
        recent_week = weekly["revenue"].iloc[-1] if len(weekly) > 0 else 0
        trend = "📈 Up" if recent_week > avg_weekly else "📉 Down" if recent_week < avg_weekly else "➡️ Flat"

        # Top categories
        cats = conn.execute("""
            SELECT p.category, ROUND(SUM(o.quantity * CAST(o.unit_price AS DOUBLE)), 2) AS rev
            FROM orders o JOIN products p ON o.product_id = p.product_id
            WHERE o.quantity > 0 AND TRY_CAST(o.unit_price AS DOUBLE) IS NOT NULL
            GROUP BY p.category ORDER BY rev DESC
        """).fetchall()

        sections.append(f"## 📊 Executive Summary")
        sections.append(f"**Period:** {len(weekly)} weeks of data")
        sections.append(f"")
        sections.append(f"### Revenue")
        sections.append(f"- **Total Revenue:** ${rev:,.2f}")
        sections.append(f"- **Avg Weekly Revenue:** ${avg_weekly:,.2f}")
        sections.append(f"- **Latest Week:** ${recent_week:,.2f} ({trend})")
        sections.append(f"")
        sections.append(f"### Data Quality")
        sections.append(f"- **Total Orders:** {total_orders:,}")
        sections.append(f"- **Invalid Quantities:** {bad_qty} ({100*bad_qty/total_orders:.1f}%)")
        sections.append(f"- **Invalid Prices:** {bad_price} ({100*bad_price/total_orders:.1f}%)")
        sections.append(f"")
        sections.append(f"### Top Categories")
        for cat, r in cats[:5]:
            sections.append(f"- **{cat}:** ${r:,.2f}")
        sections.append(f"")

        # Customers
        total_cust = conn.execute("SELECT COUNT(DISTINCT customer_id) FROM orders").fetchone()[0]
        repeat = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT customer_id, COUNT(*) AS cnt
                FROM orders GROUP BY customer_id HAVING cnt > 1
            )
        """).fetchone()[0]
        sections.append(f"### Customers")
        sections.append(f"- **Active Customers:** {total_cust:,}")
        sections.append(f"- **Repeat Customers:** {repeat:,}")
        sections.append(f"- **Repeat Rate:** {100*repeat/total_cust:.1f}%" if total_cust > 0 else "N/A")

        sections.append(f"")
        sections.append(f"---")
        sections.append(f"*Report generated by DataEng Copilot Analytics Agent*")

        return {
            "summary": f"Executive summary: ${rev:,.0f} total revenue, {total_cust:,} customers, {total_orders:,} orders",
            "detail": "\n".join(sections),
            "tokens_in": 500,
            "tokens_out": len("\n".join(sections)),
        }

    def _kpi_report(self, conn) -> dict:
        """Generate a KPI dashboard report."""
        metrics = conn.execute("""
            SELECT
              COUNT(DISTINCT o.customer_id) AS active_customers,
              COUNT(*) AS total_orders,
              ROUND(AVG(o.quantity * CAST(o.unit_price AS DOUBLE)), 2) AS avg_order_value,
              ROUND(SUM(o.quantity * CAST(o.unit_price AS DOUBLE)), 2) AS total_revenue
            FROM orders o
            WHERE o.quantity > 0 AND TRY_CAST(o.unit_price AS DOUBLE) IS NOT NULL
        """).fetchone()

        detail = (
            "## 📈 KPI Dashboard\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Active Customers | {metrics[0]:,} |\n"
            f"| Total Orders | {metrics[1]:,} |\n"
            f"| Avg Order Value | ${metrics[2]:.2f} |\n"
            f"| Total Revenue | ${metrics[3]:,.2f} |\n"
        )
        return {
            "summary": f"KPI report: ${metrics[3]:,.0f} revenue, ${metrics[2]:.2f} AOV, {metrics[0]:,} customers",
            "detail": detail,
            "tokens_in": 200,
            "tokens_out": len(detail),
        }

    def _roi_report(self, conn) -> dict:
        """Generate ROI/performance report."""
        detail = (
            "## 💰 ROI Analysis\n\n"
            "### Assumptions\n"
            "- 8-person data engineering team\n"
            "- $120,000 avg loaded cost/engineer\n"
            "- 50% automatable task-hours\n"
            "- 6-month adoption ramp\n\n"
            "### Projected Impact\n"
            "- **FTE equivalent freed:** 4.0\n"
            "- **Annual savings at ramp:** $480,000\n"
            "- **Payback period:** 6 months\n"
            "- **Year-1 net value:** $280,000\n\n"
            "### Framing\n"
            "Same productivity, reduced headcount cost — "
            "or same team shipping multiples more output."
        )
        return {
            "summary": "ROI analysis: 4 FTE freed, $480K/yr savings, 6-month payback",
            "detail": detail,
            "tokens_in": 200,
            "tokens_out": len(detail),
        }

    def _segmentation(self, conn) -> dict:
        """Customer/category segmentation report."""
        sections = []

        # By country
        countries = conn.execute("""
            SELECT c.country, COUNT(*) AS orders,
                   ROUND(SUM(o.quantity * CAST(o.unit_price AS DOUBLE)), 2) AS revenue
            FROM orders o JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.quantity > 0 AND TRY_CAST(o.unit_price AS DOUBLE) IS NOT NULL
            GROUP BY c.country ORDER BY revenue DESC
        """).fetchall()

        sections.append("## 🌍 Revenue by Country")
        sections.append("| Country | Orders | Revenue |")
        sections.append("|---------|--------|---------|")
        for country, orders, rev in countries:
            sections.append(f"| {country} | {orders:,} | ${rev:,.2f} |")

        # By channel
        channels = conn.execute("""
            SELECT channel, COUNT(*) AS orders,
                   ROUND(SUM(quantity * CAST(unit_price AS DOUBLE)), 2) AS revenue
            FROM orders
            WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
            GROUP BY channel ORDER BY revenue DESC
        """).fetchall()

        sections.append("")
        sections.append("## 📱 Revenue by Channel")
        sections.append("| Channel | Orders | Revenue |")
        sections.append("|---------|--------|---------|")
        for ch, orders, rev in channels:
            sections.append(f"| {ch} | {orders:,} | ${rev:,.2f} |")

        detail = "\n".join(sections)
        return {
            "summary": f"Segmentation: {len(countries)} countries, {len(channels)} channels",
            "detail": detail,
            "tokens_in": 300,
            "tokens_out": len(detail),
        }

    # ── Charts ───────────────────────────────────────────────

    def _build_charts(self, conn, query: str) -> list[dict]:
        charts = []
        # Monthly revenue trend
        monthly = conn.execute("""
            SELECT strftime(CAST(order_date AS DATE), '%Y-%m') AS month,
                   ROUND(SUM(quantity * CAST(unit_price AS DOUBLE)), 2) AS revenue
            FROM orders WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
            GROUP BY month ORDER BY month
        """).fetchdf()
        if len(monthly) > 0:
            charts.append({
                "type": "chart", "title": "Monthly Revenue Trend",
                "chart_type": "line",
                "x": monthly["month"].tolist(),
                "y": monthly["revenue"].tolist(),
                "x_label": "Month", "y_label": "Revenue ($)",
            })

        # Top categories
        cats = conn.execute("""
            SELECT p.category, ROUND(SUM(o.quantity * CAST(o.unit_price AS DOUBLE)), 2) AS rev
            FROM orders o JOIN products p ON o.product_id = p.product_id
            WHERE o.quantity > 0 AND TRY_CAST(o.unit_price AS DOUBLE) IS NOT NULL
            GROUP BY p.category ORDER BY rev DESC
        """).fetchdf()
        if len(cats) > 0:
            charts.append({
                "type": "chart", "title": "Revenue by Category",
                "chart_type": "bar",
                "x": cats["category"].tolist(),
                "y": cats["rev"].tolist(),
                "x_label": "Category", "y_label": "Revenue ($)",
            })

        return charts

    def cleanup(self):
        if self._conn:
            self._conn.close()
            self._conn = None