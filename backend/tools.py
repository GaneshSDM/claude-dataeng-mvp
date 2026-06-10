"""Local tools Claude drives. All execute against per-job DuckDB over sample CSVs."""
from pathlib import Path
import duckdb
from backend.anomaly_wrapper import scan_orders

TOOL_DEFS = [
    {"name": "profile_data",
     "description": "Profile a raw table: row count, columns/dtypes, null counts, 5 sample rows. Call once per raw table (orders, customers, products) before transforming.",
     "input_schema": {"type": "object", "properties": {
         "table": {"type": "string", "enum": ["orders", "customers", "products"]}},
         "required": ["table"]}},
    {"name": "run_sql",
     "description": "Execute DuckDB SQL. Use CREATE TABLE ... AS for cleaning/marts (purpose=etl); SELECT for business questions (purpose=question) or quick checks (purpose=inspect). SELECT output truncated to 50 rows.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "purpose": {"type": "string", "enum": ["etl", "question", "inspect"]}},
         "required": ["query", "purpose"]}},
    {"name": "run_anomaly_scan",
     "description": "Run the 5-layer statistical anomaly detector (pct/z-score/IQR/CUSUM/Bayesian) on weekly order GDV and volume. Call after ETL. Returns top findings.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "create_chart",
     "description": "Create a chart artifact for the executive report. Provide parallel x (labels) and y (numbers) arrays.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"},
         "chart_type": {"type": "string", "enum": ["bar", "line"]},
         "x": {"type": "array", "items": {"type": "string"}},
         "y": {"type": "array", "items": {"type": "number"}},
         "x_label": {"type": "string"}, "y_label": {"type": "string"}},
         "required": ["title", "chart_type", "x", "y"]}},
    {"name": "write_report",
     "description": "Write the final executive summary as markdown. Call exactly once, last.",
     "input_schema": {"type": "object", "properties": {"markdown": {"type": "string"}},
                      "required": ["markdown"]}},
]

STEP_FOR_TOOL = {"profile_data": "profile", "run_anomaly_scan": "anomaly", "write_report": "report"}


def step_for(tool: str, tool_input: dict) -> str:
    if tool == "run_sql":
        return {"etl": "etl", "question": "sql", "inspect": "profile"}[tool_input.get("purpose", "inspect")]
    if tool == "create_chart":
        return "report"
    return STEP_FOR_TOOL.get(tool, "")


class ToolExecutor:
    def __init__(self, data_dir: Path):
        self.conn = duckdb.connect(":memory:")
        for t in ("orders", "customers", "products"):
            self.conn.execute(
                f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{(Path(data_dir) / t).as_posix()}.csv')")

    def execute(self, name: str, tool_input: dict) -> dict:
        """Returns {"result": <json-able>, "artifact": <dict|None>}."""
        if name == "profile_data":
            t = tool_input["table"]
            df = self.conn.execute(f"SELECT * FROM {t}").fetchdf()
            return {"result": {
                "row_count": len(df),
                "columns": {c: str(df[c].dtype) for c in df.columns},
                "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
                "sample_rows": df.head(5).astype(str).to_dict("records")}, "artifact": None}
        if name == "run_sql":
            cur = self.conn.execute(tool_input["query"])
            if cur.description:
                rows = cur.fetchmany(50)
                return {"result": {"columns": [d[0] for d in cur.description],
                                   "rows": [list(r) for r in rows]}, "artifact": None}
            return {"result": {"status": "ok"}, "artifact": None}
        if name == "run_anomaly_scan":
            return {"result": {"findings": scan_orders(self.conn)}, "artifact": None}
        if name == "create_chart":
            art = {"type": "chart", "title": tool_input["title"],
                   "chart_type": tool_input["chart_type"],
                   "x": tool_input["x"], "y": tool_input["y"],
                   "x_label": tool_input.get("x_label", ""), "y_label": tool_input.get("y_label", "")}
            return {"result": {"status": "chart created", "title": art["title"]}, "artifact": art}
        if name == "write_report":
            return {"result": {"status": "report saved"},
                    "artifact": {"type": "report", "markdown": tool_input["markdown"]}}
        raise ValueError(f"unknown tool {name}")
