"""Unified MedallionPipeline — runs Bronze -> Silver -> Gold in sequence."""
from __future__ import annotations
from pathlib import Path
import duckdb
from core.medallion.bronze import ingest_bronze
from core.medallion.silver import run_silver
from core.medallion.gold import build_gold_marts


class MedallionPipeline:
    """Run the full medallion pipeline: Bronze (raw) -> Silver (clean) -> Gold (marts)."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def run(self, conn: duckdb.DuckDBPyConnection | None = None) -> dict:
        """Execute Bronze -> Silver -> Gold pipeline. Returns full report."""
        close_conn = conn is None
        conn = conn or duckdb.connect(":memory:")
        try:
            bronze = ingest_bronze(conn, self.data_dir)
            silver = run_silver(conn)
            gold = build_gold_marts(conn)
            return {"status": "ok", "bronze": bronze, "silver": silver, "gold": gold}
        finally:
            if close_conn:
                conn.close()