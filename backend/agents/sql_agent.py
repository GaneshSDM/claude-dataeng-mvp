"""SQL Agent — translates natural language to DuckDB SQL and executes it."""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from backend.agents.base import BaseAgent, AgentResult


class SQLAgent(BaseAgent):
    """Converts natural-language data questions into SQL, runs them, returns results + optional charts."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def name(self) -> str:
        return "sql"

    @property
    def description(self) -> str:
        return "Answer data questions with SQL — ask anything about orders, customers, products"

    @property
    def routing_hints(self) -> list[str]:
        return ["sql", "query", "question", "select", "how many", "what is", "show me",
                "revenue", "top", "average", "count", "trend", "monthly", "weekly"]

    # ── helpers ──────────────────────────────────────────────

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
            for t in ("orders", "customers", "products"):
                path = (self.data_dir / t).as_posix() + ".csv"
                self._conn.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{path}')")
        return self._conn

    def _schema_hint(self) -> str:
        conn = self._get_conn()
        lines = []
        for t in ("orders", "customers", "products"):
            cols = conn.execute(f"DESCRIBE {t}").fetchall()
            lines.append(f"  {t}: " + ", ".join(f"{c[0]} {c[1]}" for c in cols))
        return "\n".join(lines)

    # ── core ─────────────────────────────────────────────────

    def run(self, user_input: str, context: Optional[dict] = None) -> AgentResult:
        t0 = time.time()
        conn = self._get_conn()
        schema = self._schema_hint()
        has_chart = any(w in user_input.lower() for w in ["chart", "plot", "visualize", "trend"])

        # Build a prompt for Claude or a rule-based SQL generator
        # For demo we use a lightweight heuristic + fallback pattern matcher
        sql = self._generate_sql(user_input, schema)
        if not sql:
            return AgentResult(
                agent="sql", status="error",
                summary="Could not generate SQL from your question.",
                detail=f"Schema:\n{schema}\n\nQuestion: {user_input}",
                error="SQL generation failed",
                duration_s=time.time() - t0,
            )

        try:
            df = conn.execute(sql).fetchdf()
            duration = time.time() - t0
            rows_count = len(df)
            truncated = df.head(50)

            artifacts = []
            detail_parts = [f"**Generated SQL:**\n```sql\n{sql}\n```"]

            if rows_count > 0:
                detail_parts.append(f"\n**Results** ({rows_count} rows, showing {min(50, rows_count)}):")
                detail_parts.append(truncated.to_markdown(index=False))

                if has_chart and len(df.columns) >= 2:
                    chart_art = self._make_chart_artifact(df, user_input)
                    if chart_art:
                        artifacts.append(chart_art)
            else:
                detail_parts.append("\n_Query returned no rows._")

            return AgentResult(
                agent="sql",
                status="ok",
                summary=f"SQL query returned {rows_count} rows",
                detail="\n".join(detail_parts),
                sql=sql,
                artifacts=artifacts,
                tokens_in=len(user_input) + len(schema),
                tokens_out=len(sql) + rows_count * 20,
                cost_usd=0.01,
                duration_s=duration,
            )
        except Exception as e:
            return AgentResult(
                agent="sql", status="error",
                summary="SQL execution failed",
                detail=f"SQL:\n```sql\n{sql}```\n\nError: {e}",
                error=str(e),
                duration_s=time.time() - t0,
            )

    # ── SQL generation (heuristic + pattern matching) ────────

    def _generate_sql(self, question: str, schema: str) -> str | None:
        q = question.lower()

        # --- Top products by revenue ---
        if re.search(r"top\s+(\d+\s+)?product", q) and "revenue" in q:
            m = re.search(r"top\s+(\d+)", q)
            n = m.group(1) if m else "5"
            return f"""SELECT p.product_name, p.category,
                              ROUND(SUM(o.quantity * CAST(o.unit_price AS DOUBLE)), 2) AS revenue
                       FROM orders o JOIN products p ON o.product_id = p.product_id
                       WHERE o.quantity > 0 AND TRY_CAST(o.unit_price AS DOUBLE) IS NOT NULL
                       GROUP BY p.product_name, p.category
                       ORDER BY revenue DESC LIMIT {n}"""

        # --- Monthly revenue trend ---
        if re.search(r"monthly.*(revenue|trend|sales)", q):
            return """SELECT strftime(CAST(order_date AS DATE), '%Y-%m') AS month,
                             ROUND(SUM(quantity * CAST(unit_price AS DOUBLE)), 2) AS revenue,
                             COUNT(*) AS order_count
                      FROM orders
                      WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                      GROUP BY month ORDER BY month"""

        # --- Repeat customer rate ---
        if "repeat" in q or "returning" in q or "customer.*rate" in q:
            return """WITH cust_orders AS (
                        SELECT customer_id, COUNT(*) AS cnt
                        FROM orders GROUP BY customer_id
                      )
                      SELECT ROUND(100.0 * SUM(CASE WHEN cnt > 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS repeat_customer_pct,
                             COUNT(*) AS total_customers,
                             SUM(CASE WHEN cnt > 1 THEN 1 ELSE 0 END) AS repeat_customers
                      FROM cust_orders"""

        # --- Revenue by channel ---
        if re.search(r"(revenue|sales).*(channel|by channel)", q):
            return """SELECT channel,
                             ROUND(SUM(quantity * CAST(unit_price AS DOUBLE)), 2) AS revenue,
                             COUNT(*) AS orders
                      FROM orders
                      WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                      GROUP BY channel ORDER BY revenue DESC"""

        # --- Revenue by country ---
        if re.search(r"(revenue|sales).*(country|by country)", q):
            return """SELECT c.country,
                             ROUND(SUM(o.quantity * CAST(o.unit_price AS DOUBLE)), 2) AS revenue,
                             COUNT(*) AS orders
                      FROM orders o JOIN customers c ON o.customer_id = c.customer_id
                      WHERE o.quantity > 0 AND TRY_CAST(o.unit_price AS DOUBLE) IS NOT NULL
                      GROUP BY c.country ORDER BY revenue DESC"""

        # --- Weekly trend ---
        if re.search(r"weekly.*(revenue|trend|sales)", q):
            return """SELECT date_trunc('week', CAST(order_date AS DATE)) AS week,
                             ROUND(SUM(quantity * CAST(unit_price AS DOUBLE)), 2) AS revenue,
                             COUNT(*) AS order_count
                      FROM orders
                      WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                      GROUP BY week ORDER BY week"""

        # --- Customer count / total customers ---
        if re.search(r"(how many|total|number of).*customers", q):
            return "SELECT COUNT(DISTINCT customer_id) AS total_customers FROM customers"

        # --- Product count ---
        if re.search(r"(how many|total|number of).*products", q):
            return "SELECT COUNT(*) AS total_products FROM products"

        # --- Total orders ---
        if re.search(r"(how many|total|number of).*orders", q):
            return "SELECT COUNT(*) AS total_orders FROM orders"

        # --- Average order value ---
        if re.search(r"average.*order.*(value|amount)", q):
            return """SELECT ROUND(AVG(order_total), 2) AS avg_order_value
                      FROM (SELECT order_id, SUM(quantity * CAST(unit_price AS DOUBLE)) AS order_total
                            FROM orders
                            WHERE quantity > 0 AND TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                            GROUP BY order_id)"""

        # --- Fallback: generic select ---
        # For unrecognized questions, do a simple exploration
        if re.search(r"(describe|schema|columns|structure|what.*tables)", q):
            return self._generic_describe()

        return None

    def _generic_describe(self) -> str:
        """Return schema description."""
        return "SELECT table_name, column_name, data_type FROM (DESCRIBE orders) UNION ALL (DESCRIBE customers) UNION ALL (DESCRIBE products)"

    # ── chart helper ─────────────────────────────────────────

    def _make_chart_artifact(self, df: pd.DataFrame, question: str) -> dict | None:
        if len(df.columns) < 2:
            return None
        # Auto-detect: first text col = labels, first numeric col = values
        label_col = None
        value_col = None
        for c in df.columns:
            if df[c].dtype in ("object", "string") or "date" in str(c).lower():
                label_col = label_col or c
            elif df[c].dtype in ("int64", "float64"):
                value_col = value_col or c
        if not (label_col and value_col):
            x_col, y_col = df.columns[0], df.columns[1]
        else:
            x_col, y_col = label_col, value_col

        chart_type = "line" if any(w in question.lower() for w in ["trend", "monthly", "weekly", "over time"]) else "bar"
        return {
            "type": "chart",
            "title": question[:60],
            "chart_type": chart_type,
            "x": df[x_col].astype(str).head(30).tolist(),
            "y": pd.to_numeric(df[y_col], errors="coerce").dropna().head(30).tolist(),
            "x_label": x_col,
            "y_label": y_col,
        }

    def cleanup(self):
        if self._conn:
            self._conn.close()
            self._conn = None