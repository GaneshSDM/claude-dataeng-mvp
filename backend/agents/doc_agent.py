"""Documentation Agent — generates data dictionaries, pipeline docs, and markdown reports."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from core.sql_engine.lineage import extract_lineage

from backend.agents.base import BaseAgent, AgentResult


class DocumentationAgent(BaseAgent):
    """Generates data dictionaries, pipeline documentation, and structured reports."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def name(self) -> str:
        return "docs"

    @property
    def description(self) -> str:
        return "Generate data dictionaries, pipeline docs, and schema documentation"

    @property
    def routing_hints(self) -> list[str]:
        return ["doc", "documentation", "dictionary", "schema", "describe", "data catalog",
                "lineage", "metadata", "docs"]

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

        if "dictionary" in q or "data catalog" in q:
            doc = self._data_dictionary(conn)
        elif "pipeline" in q or "lineage" in q:
            doc = self._pipeline_documentation(conn)
        else:
            doc = self._schema_documentation(conn)

        return AgentResult(
            agent="docs",
            status="ok",
            summary=doc["summary"],
            detail=doc["detail"],
            tokens_in=300,
            tokens_out=len(doc["detail"]),
            cost_usd=0.01,
            duration_s=time.time() - t0,
        )

    def _data_dictionary(self, conn) -> dict:
        """Generate detailed data dictionary with column descriptions."""
        sections = []
        for table in ("orders", "customers", "products"):
            cols = conn.execute(f"DESCRIBE {table}").fetchall()
            sample = conn.execute(f"SELECT * FROM {table} LIMIT 3").fetchdf()
            nulls = {c: int(sample[c].isna().sum()) for c in sample.columns}

            sections.append(f"## 📋 `{table}` Table")
            sections.append("")
            sections.append("| Column | Type | Nulls in Sample | Description |")
            sections.append("|--------|------|----------------|-------------|")
            for col in cols:
                name, dtype = col[0], col[1]
                null_count = nulls.get(name, "?")
                desc = self._column_description(table, name)
                sections.append(f"| `{name}` | {dtype} | {null_count} | {desc} |")
            sections.append("")

        detail = "\n".join(sections)
        return {
            "summary": "Data dictionary generated for orders, customers, and products",
            "detail": detail,
        }

    def _pipeline_documentation(self, conn) -> dict:
        """Generate pipeline documentation with lineage."""
        # Extract lineage from key SQL patterns
        lineage_samples = []
        for sql_pattern, label in [
            ("SELECT o.order_id, o.customer_id, o.quantity, o.unit_price * o.quantity AS revenue FROM orders o", "orders → revenue calc"),
            ("SELECT c.customer_id, c.name, c.country, COUNT(o.order_id) AS order_count FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name, c.country", "customers → customer metrics"),
        ]:
            try:
                lineage = extract_lineage(sql_pattern)
                lineage_samples.append({"label": label, "lineage": lineage})
            except Exception:
                pass

        lineage_md = ""
        if lineage_samples:
            lineage_md = "\n### Column-Level Lineage\n"
            for sample in lineage_samples:
                lineage_md += f"\n**{sample['label']}**\n"
                for col in sample["lineage"][:6]:
                    sources = ", ".join(f"{s['table']}.{s['column']}" for s in col.get("sources", []))
                    lineage_md += f"- `{col['column']}` ← {sources or 'direct'} ({col['transformation_type']})\n"

        detail = (
            "## 🔄 Data Pipeline Documentation\n\n"
            "### Architecture: Medallion (Bronze → Silver → Gold)\n\n"
            "### Source Tables\n"
            "- **orders** — Raw order transactions (78 weeks)\n"
            "- **customers** — Customer master data\n"
            "- **products** — Product catalog with categories\n\n"
            "### Bronze Layer\n"
            "- Raw CSV ingestion with schema tracking\n"
            "- Schema change detection via hash comparison\n\n"
            "### Silver Layer\n"
            "- Clean negative quantities → NULL\n"
            "- Parse non-numeric prices → NULL\n"
            "- Deduplicate customer records\n"
            "- Validate product categories\n\n"
            "### Gold Layer\n"
            "- `weekly_sales`: Revenue, order counts, unique customers by week\n"
            "- `customer_metrics`: Lifetime value, order frequency\n"
            "- `category_performance`: Revenue by product category\n\n"
            "### Quality Checks\n"
            "- 15-category PII scanning\n"
            "- 19-rule SQL anti-pattern detection\n"
            "- 5-layer statistical anomaly detection\n"
            "- Null & duplicate analysis\n\n"
            "### Agents\n"
            "- **SQL Agent**: Natural language → SQL + anti-pattern validation\n"
            "- **ETL Agent**: Medallion pipeline (bronze → silver → gold)\n"
            "- **Quality Agent**: Anomaly detection, 15-category PII, quality checks\n"
            "- **Analytics Agent**: Reports, charts, KPIs, semantic metrics\n"
            "- **Documentation Agent**: Data dictionary, lineage, pipeline docs\n"
            "- **Orchestrator**: Coordinates multi-agent workflows\n"
            f"{lineage_md}"
        )
        return {
            "summary": "Pipeline docs: bronze → silver → gold with lineage tracking",
            "detail": detail,
        }

    def _schema_documentation(self, conn) -> dict:
        """Generate schema documentation."""
        sections = []
        for table in ("orders", "customers", "products"):
            cols = conn.execute(f"DESCRIBE {table}").fetchall()
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            sections.append(
                f"### `{table}` ({row_count:,} rows)\n"
                + "\n".join(f"- `{c[0]}` ({c[1]})" for c in cols)
            )
        detail = "## 📐 Schema Documentation\n\n" + "\n\n".join(sections)
        return {
            "summary": "Schema docs for 3 tables",
            "detail": detail,
        }

    def _column_description(self, table: str, column: str) -> str:
        """Provide a human-readable description for known columns."""
        descriptions = {
            "orders": {
                "order_id": "Unique order identifier",
                "customer_id": "Foreign key to customers",
                "product_id": "Foreign key to products",
                "order_date": "Date the order was placed",
                "quantity": "Number of units ordered",
                "unit_price": "Price per unit (may contain dirty data)",
                "channel": "Sales channel (web, mobile, partner)",
            },
            "customers": {
                "customer_id": "Unique customer identifier",
                "name": "Customer display name",
                "email": "Email address (may contain nulls)",
                "country": "Country code (US, IN, DE, UK)",
                "signup_date": "Date of customer registration",
            },
            "products": {
                "product_id": "Unique product identifier",
                "product_name": "Product display name",
                "category": "Product category (Electronics, Home, Toys, Office)",
                "list_price": "Standard list price",
            },
        }
        return descriptions.get(table, {}).get(column, "—")

    def cleanup(self):
        if self._conn:
            self._conn.close()
            self._conn = None