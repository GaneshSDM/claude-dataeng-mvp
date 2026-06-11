"""ETL Agent — profiles raw data, performs cleaning/transformations, builds marts."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import duckdb

from backend.agents.base import BaseAgent, AgentResult
from core.medallion.pipeline import MedallionPipeline


class ETLAgent(BaseAgent):
    """Runs ETL lifecycle via the Medallion pipeline (bronze/silver/gold)."""

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
        artifacts = []
        detail_parts = []

        # Use the Medallion pipeline instead of ad-hoc methods
        pipeline = MedallionPipeline(self.data_dir)
        result = pipeline.run(conn)

        if result["status"] != "ok":
            return AgentResult(
                agent="etl", status="error",
                summary="Medallion pipeline failed",
                detail=str(result),
                duration_s=time.time() - t0,
            )

        # Format report
        bronze = result["bronze"]
        silver = result["silver"]
        gold = result["gold"]

        # Bronze summary
        bronze_lines = []
        for table, info in bronze.items():
            bronze_lines.append(f"- **{table}**: {info['row_count']:,} rows, {len(info['columns'])} columns")
        detail_parts.append("### Bronze Layer (Raw Ingestion)\n" + "\n".join(bronze_lines))

        # Silver summary
        silver_detail = "### Silver Layer (Cleaning)\n" + "\n".join(f"- {s}" for s in silver["steps"])
        if silver["issues"]:
            silver_detail += "\n\n**Issues:**\n" + "\n".join(f"- Warning: {i}" for i in silver["issues"])
        detail_parts.append(silver_detail)

        # Gold summary
        gold_lines = []
        for mart, info in gold.items():
            line = f"- **{mart}**: {info['row_count']:,} rows"
            if "total_revenue" in info:
                line += f", revenue=${info['total_revenue']:,.2f}"
            gold_lines.append(line)
        detail_parts.append("### Gold Layer (Analytics Marts)\n" + "\n".join(gold_lines))

        silver_rows = sum(silver["row_counts"].values())
        gold_rows = sum(v["row_count"] for v in gold.values())
        summary = f"Medallion pipeline: {bronze['bronze_orders']['row_count']:,} raw -> {silver_rows:,} clean -> {gold_rows:,} mart rows"

        return AgentResult(
            agent="etl",
            status="ok",
            summary=summary,
            detail="\n\n".join(detail_parts),
            tokens_in=500,
            tokens_out=len(summary),
            cost_usd=0.02,
            duration_s=time.time() - t0,
        )

    def cleanup(self):
        if self._conn:
            self._conn.close()
            self._conn = None