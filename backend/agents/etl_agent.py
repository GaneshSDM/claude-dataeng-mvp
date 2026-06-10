"""ETL Agent — profiles raw data, performs cleaning/transformations, builds marts."""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from backend.agents.base import BaseAgent, AgentResult


class ETLAgent(BaseAgent):
    """Runs the ETL lifecycle: profile raw tables, clean data, build analytics marts."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def name(self) -> str:
        return "etl"

    @property
    def description(self) -> str:
        return "Profile, clean, and transform data — build analytics-ready tables from raw sources"

    @property
    def routing_hints(self) -> list[str]:
        return ["etl", "pipeline", "transform", "clean", "profile", "mart", "data quality"]

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
        detail_parts = []
        total_tokens_in = 0
        total_tokens_out = 0

        # Always profile if asked or if "full" lifecycle
        if any(w in q for w in ["profile", "full", "all", "complete"]):
            prof = self._profile(conn, detail_parts)
            total_tokens_in += prof[0]
            total_tokens_out += prof[1]

        # Clean if asked
        if any(w in q for w in ["clean", "full", "all", "complete", "etl"]):
            cln = self._clean(conn, detail_parts)
            total_tokens_in += cln[0]
            total_tokens_out += cln[1]

        # Build marts
        if any(w in q for w in ["mart", "full", "all", "complete", "etl", "transform", "weekly"]):
            mrt = self._build_marts(conn, detail_parts)
            total_tokens_in += mrt[0]
            total_tokens_out += mrt[1]

        if not detail_parts:
            # Default: run full lifecycle
            self._profile(conn, detail_parts)
            self._clean(conn, detail_parts)
            self._build_marts(conn, detail_parts)

        summary = f"ETL complete — profiled, cleaned, and built analytics marts"
        return AgentResult(
            agent="etl",
            status="ok",
            summary=summary,
            detail="\n\n".join(detail_parts),
            artifacts=artifacts,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            cost_usd=0.02,
            duration_s=time.time() - t0,
        )

    def _profile(self, conn, detail_parts) -> tuple[int, int]:
        """Profile raw tables."""
        ti, to = 0, 0
        for t in ("orders", "customers", "products"):
            df = conn.execute(f"SELECT * FROM {t}").fetchdf()
            nulls = {c: int(df[c].isna().sum()) for c in df.columns}
            sample = df.head(5).to_dict("records")
            ti += len(df) * 5
            to += len(str(sample))
            detail_parts.append(
                f"### 📊 {t}\n"
                f"- **Rows:** {len(df)} | **Columns:** {len(df.columns)}\n"
                f"- **Nulls:** {nulls}\n"
                f"- **Sample:**\n```json\n{json.dumps(sample, default=str, indent=2)[:1000]}\n```"
            )
        return ti, to

    def _clean(self, conn, detail_parts) -> tuple[int, int]:
        """Perform data cleaning."""
        steps = []

        # Clean orders: remove negative quantity, bad prices
        conn.execute("""
            CREATE OR REPLACE TABLE orders_clean AS
            SELECT *,
                   CASE WHEN quantity > 0 THEN quantity ELSE NULL END AS qty_clean,
                   CASE WHEN TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                        THEN CAST(unit_price AS DOUBLE) ELSE NULL END AS price_clean
            FROM orders
        """)
        bad_qty = conn.execute("SELECT COUNT(*) FROM orders WHERE quantity <= 0").fetchone()[0]
        bad_price = conn.execute("SELECT COUNT(*) FROM orders WHERE TRY_CAST(unit_price AS DOUBLE) IS NULL").fetchone()[0]
        steps.append(f"- Orders with negative qty: {bad_qty} → cleaned to NULL")
        steps.append(f"- Orders with non-numeric prices: {bad_price} → cleaned to NULL")
        total = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        df = conn.execute("SELECT * FROM orders_clean WHERE qty_clean IS NOT NULL AND price_clean IS NOT NULL").fetchdf()

        # Deduplicate customers
        conn.execute("""
            CREATE OR REPLACE TABLE customers_clean AS
            SELECT DISTINCT * FROM customers
        """)
        orig_cust = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        deduped = conn.execute("SELECT COUNT(*) FROM customers_clean").fetchone()[0]
        dupes = orig_cust - deduped
        if dupes:
            steps.append(f"- Customers deduplicated: {dupes} duplicate rows removed")

        detail_parts.append(
            f"### 🧹 Data Cleaning\n"
            f"- **Total orders:** {total}\n" + "\n".join(steps) + "\n"
            f"- **Clean order rows:** {len(df)}"
        )
        return total, len(df)

    def _build_marts(self, conn, detail_parts) -> tuple[int, int]:
        """Build analytics marts."""
        conn.execute("""
            CREATE OR REPLACE TABLE weekly_sales AS
            SELECT date_trunc('week', CAST(order_date AS DATE)) AS week,
                   ROUND(SUM(qty_clean * price_clean), 2) AS revenue,
                   COUNT(*) AS order_count,
                   COUNT(DISTINCT customer_id) AS unique_customers
            FROM orders_clean
            WHERE qty_clean IS NOT NULL AND price_clean IS NOT NULL
            GROUP BY week ORDER BY week
        """)
        ws = conn.execute("SELECT * FROM weekly_sales").fetchdf()
        total_rev = ws["revenue"].sum()
        avg_rev = ws["revenue"].mean()
        detail_parts.append(
            f"### 📈 Weekly Sales Mart\n"
            f"- **Weeks:** {len(ws)}\n"
            f"- **Total revenue:** ${total_rev:,.2f}\n"
            f"- **Avg weekly revenue:** ${avg_rev:,.2f}\n"
            f"- **Schema:** {', '.join(f'{c}' for c in ws.columns)}"
        )
        return len(ws), 0

    def cleanup(self):
        if self._conn:
            self._conn.close()
            self._conn = None