"""Bronze layer — raw ingestion with schema tracking and change detection."""
from __future__ import annotations
from pathlib import Path
import duckdb
import json


def ingest_bronze(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> dict:
    """Load raw CSVs into bronze tables with schema tracking.
    Returns manifest: {table_name: {row_count, columns, null_counts, schema_hash}}
    """
    manifest = {}
    for table in ("orders", "customers", "products"):
        path = (data_dir / table).as_posix() + ".csv"
        conn.execute(f"CREATE OR REPLACE TABLE bronze_{table} AS SELECT * FROM read_csv_auto('{path}')")
        df = conn.execute(f"SELECT * FROM bronze_{table}").fetchdf()
        cols = conn.execute(f"DESCRIBE bronze_{table}").fetchall()
        nulls = {c[0]: int(df[c[0]].isna().sum()) for c in cols}
        schema_sig = json.dumps({c[0]: c[1] for c in cols}, sort_keys=True)
        manifest[f"bronze_{table}"] = {
            "row_count": len(df),
            "columns": {c[0]: c[1] for c in cols},
            "null_counts": nulls,
            "schema_hash": hash(schema_sig),
        }
    return manifest