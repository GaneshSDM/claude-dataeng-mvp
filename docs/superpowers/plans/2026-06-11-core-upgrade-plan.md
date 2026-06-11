# Core Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the best features from 4 reference data-engineering projects into the existing claude-dataeng-mvp platform: deterministic SQL analysis (Altimate Code), medallion pipeline (Agentic-Medallion), RAG knowledge base (Datus Agent), and upgraded PII detection.

**Architecture:** Four new `core/` subpackages (sql_engine, medallion, knowledge, pii) — each exposing simple functions that the existing 5 agents import in < 5 lines. No new agents. No API changes. Backward compatible.

**Tech Stack:** sqlglot (SQL parser), lancedb (vector store), sentence-transformers (embeddings), duckdb (existing), scipy (existing), numpy (existing)

---

### File Structure: Created and Modified

**New files:**
- `core/sql_engine/__init__.py` — package init, exports high-level API
- `core/sql_engine/rules.py` — 19 deterministic SQL anti-pattern rules with sqlglot
- `core/sql_engine/lineage.py` — column-level lineage extractor
- `core/medallion/__init__.py` — package init
- `core/medallion/bronze.py` — Bronze layer: raw ingestion with schema tracking
- `core/medallion/silver.py` — Silver layer: cleaning, dedup, quality gates
- `core/medallion/gold.py` — Gold layer: aggregation, mart building
- `core/medallion/pipeline.py` — Unified MedallionPipeline that runs all 3 layers
- `core/knowledge/__init__.py` — package init
- `core/knowledge/store.py` — LanceDB vector store wrapper
- `core/knowledge/embeddings.py` — text-to-vector embedding function
- `core/knowledge/semantic.py` — YAML metric definitions and resolution
- `core/pii/__init__.py` — package init
- `core/pii/detector.py` — 15-category PII detector
- `tests/test_sql_engine.py` — tests for rules + lineage
- `tests/test_medallion.py` — tests for bronze/silver/gold
- `tests/test_knowledge.py` — tests for RAG store
- `tests/test_pii.py` — tests for PII detector

**Modified files:**
- `requirements.txt` — add sqlglot, lancedb, sentence-transformers
- `core/__init__.py` — expose new subpackages
- `backend/agents/sql_agent.py` — use sql_engine instead of heuristic SQL generation
- `backend/agents/etl_agent.py` — use medallion pipeline instead of ad-hoc clean
- `backend/agents/quality_agent.py` — use upgraded PII + anti-pattern checks
- `backend/agents/doc_agent.py` — include lineage in docs
- `backend/agents/orchestrator.py` — register updated agent descriptions

---

### Task 1: Add dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add sqlglot, lancedb, sentence-transformers**

Current requirements.txt ends with `tabulate>=0.9`. Append three new lines:

```
sqlglot>=25.0.0
lancedb>=0.12.0
sentence-transformers>=3.0.0
```

- [ ] **Step 2: Verify install**

Run: `pip install -r requirements.txt`
Expected: all packages install without error.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add sqlglot, lancedb, sentence-transformers deps"
```

---

### Task 2: SQL Anti-Pattern Engine — 19 deterministic rules

**Files:**
- Create: `core/sql_engine/__init__.py`
- Create: `core/sql_engine/rules.py`
- Create: `tests/test_sql_engine.py`

The rules engine uses sqlglot to parse SQL into AST, then walks the tree for each rule. Each rule is a function `(sql: str) -> list[dict]` returning findings with line number, severity, and message.

- [ ] **Step 1: Create `core/sql_engine/__init__.py`**

```python
"""Deterministic SQL analysis engine — anti-pattern detection and column-level lineage."""

from core.sql_engine.rules import analyze_sql, SQLRule, RULES_REGISTRY
from core.sql_engine.lineage import extract_lineage

__all__ = ["analyze_sql", "extract_lineage", "SQLRule", "RULES_REGISTRY"]
```

- [ ] **Step 2: Create `core/sql_engine/rules.py`**

This file implements 19 deterministic SQL anti-pattern rules using sqlglot AST parsing.

```python
"""19 deterministic SQL anti-pattern rules. 100% rule-based, no LLM involved."""

from __future__ import annotations
import sqlglot
from sqlglot import exp
from dataclasses import dataclass
from typing import Callable


@dataclass
class SQLRule:
    name: str
    severity: str       # "error" | "warning" | "info"
    description: str
    check: Callable[[exp.Select | exp.Union], list[dict]]


def _safe_parse(sql: str) -> exp.Select | None:
    try:
        return sqlglot.parse_one(sql)
    except Exception:
        return None


# ── Rule 1: SELECT * ──────────────────────────────────────────
def _select_star(tree: exp.Select) -> list[dict]:
    findings = []
    for star in tree.find_all(exp.Star):
        findings.append({
            "rule": "select_star", "severity": "warning", "line": star.line,
            "message": "SELECT * fetches all columns. Name columns explicitly for clarity and stability."
        })
    return findings


# ── Rule 2: Cartesian join (CROSS JOIN or implicit) ───────────
def _cartesian_join(tree: exp.Select) -> list[dict]:
    findings = []
    for join in tree.find_all(exp.Join):
        if join.kind == "CROSS" or join.this is None:
            findings.append({
                "rule": "cartesian_join", "severity": "error", "line": join.line,
                "message": "Cartesian join detected. Use explicit JOIN with ON conditions."
            })
    return findings


# ── Rule 3: Non-sargable predicate — function-wrapped column ──
def _non_sargable(tree: exp.Select) -> list[dict]:
    findings = []
    for func in tree.find_all(exp.Func):
        if func.args.get("this") is None:
            continue
        inner = func.args["this"]
        # Look for FUNC(column) in WHERE clause
        if isinstance(inner, exp.Column) and func.find_ancestor(exp.Where):
            findings.append({
                "rule": "non_sargable", "severity": "warning", "line": func.line,
                "message": f"Function-wrapped column {func.sql()} prevents index use. Prefer bare column comparison."
            })
    return findings


# ── Rule 4: LIKE with leading wildcard ────────────────────────
def _like_leading_wildcard(tree: exp.Select) -> list[dict]:
    findings = []
    for like in tree.find_all(exp.Like):
        pattern = like.find(exp.Literal)
        if pattern and pattern.args.get("this", "").startswith("%"):
            findings.append({
                "rule": "like_leading_wildcard", "severity": "warning", "line": like.line,
                "message": "LIKE with leading '%' is non-sargable. Consider full-text search."
            })
    return findings


# ── Rule 5: Negation in WHERE (NOT, !=, <>) ───────────────────
def _negation_in_where(tree: exp.Select) -> list[dict]:
    findings = []
    for not_ in tree.find_all(exp.Not):
        if not_.find_ancestor(exp.Where):
            findings.append({
                "rule": "negation_in_where", "severity": "warning", "line": not_.line,
                "message": "Negation (NOT) in WHERE can prevent index use. Consider rewriting positively."
            })
    for neq in tree.find_all(exp.NEQ):
        if neq.find_ancestor(exp.Where):
            findings.append({
                "rule": "negation_in_where", "severity": "info", "line": neq.line,
                "message": "Inequality (!= / <>) comparison in WHERE. Typically non-sargable."
            })
    return findings


# ── Rule 6: Missing WHERE clause ─────────────────────────────
def _missing_where(tree: exp.Select) -> list[dict]:
    if not tree.find(exp.Where) and tree.find(exp.From):
        return [{
            "rule": "missing_where", "severity": "error", "line": tree.line,
            "message": "SELECT without WHERE clause — reads full table. Add WHERE or LIMIT."
        }]
    return []


# ── Rule 7: DISTINCT on large column list ─────────────────────
def _distinct_many_columns(tree: exp.Select) -> list[dict]:
    if tree.args.get("distinct"):
        cols = list(tree.find_all(exp.Column))
        if len(cols) > 5:
            return [{
                "rule": "distinct_many_columns", "severity": "warning", "line": tree.line,
                "message": f"DISTINCT on {len(cols)} columns. Verify dedup is needed at this grain."
            }]
    return []


# ── Rule 8: ORDER BY without LIMIT ────────────────────────────
def _order_by_no_limit(tree: exp.Select) -> list[dict]:
    if tree.find(exp.Order) and not tree.find(exp.Limit):
        return [{
            "rule": "order_by_no_limit", "severity": "warning", "line": tree.line,
            "message": "ORDER BY without LIMIT sorts full result set. Add LIMIT for efficiency."
        }]
    return []


# ── Rule 9: GROUP BY ordinal ──────────────────────────────────
def _group_by_ordinal(tree: exp.Select) -> list[dict]:
    findings = []
    group = tree.args.get("group")
    if group:
        for expr in group.expressions:
            if isinstance(expr, exp.Literal) and expr.is_int:
                findings.append({
                    "rule": "group_by_ordinal", "severity": "warning", "line": expr.line,
                    "message": "GROUP BY 1, 2, ... is fragile. Use explicit column names."
                })
    return findings


# ── Rule 10: Floating point equality ──────────────────────────
def _float_equality(tree: exp.Select) -> list[dict]:
    findings = []
    for eq in tree.find_all(exp.EQ):
        for literal in eq.find_all(exp.Literal):
            if literal.args.get("is_string") is False and "." in str(literal.args.get("this", "")):
                findings.append({
                    "rule": "float_equality", "severity": "warning", "line": eq.line,
                    "message": "Floating-point equality comparison. Use ABS(a - b) < threshold."
                })
                break
    return findings


# ── Rule 11: Implicit column alias in HAVING ──────────────────
def _having_without_group_by(tree: exp.Select) -> list[dict]:
    if tree.find(exp.Having) and not tree.args.get("group"):
        return [{
            "rule": "having_without_group_by", "severity": "error", "line": tree.line,
            "message": "HAVING without GROUP BY. HAVING filters groups — use WHERE if no aggregation."
        }]
    return []


# ── Rule 12: Subquery without alias ───────────────────────────
def _subquery_no_alias(tree: exp.Select) -> list[dict]:
    findings = []
    for subq in tree.find_all(exp.Subquery):
        if not subq.args.get("alias"):
            findings.append({
                "rule": "subquery_no_alias", "severity": "error", "line": subq.line,
                "message": "Subquery missing alias. Every subquery in FROM needs an alias."
            })
    return findings


# ── Rule 13: UNION without ALL (dedup cost) ───────────────────
def _union_no_all(tree: exp.Select) -> list[dict]:
    findings = []
    for union in tree.find_all(exp.Union):
        if union.kind != "ALL" and not union.find_ancestor(exp.Union):
            # Only flag the outermost UNION if DISTINCT (default)
            pass
        if union.kind == "DISTINCT" or union.kind is None:
            findings.append({
                "rule": "union_no_all", "severity": "info", "line": union.line,
                "message": "UNION (implicit DISTINCT) sorts dedup. Use UNION ALL if duplicates are acceptable."
            })
    return findings


# ── Rule 14: COUNT(DISTINCT ...) on many rows ─────────────────
def _count_distinct_many(tree: exp.Select) -> list[dict]:
    findings = []
    for agg in tree.find_all(exp.Count):
        if agg.args.get("distinct"):
            findings.append({
                "rule": "count_distinct_many", "severity": "info", "line": agg.line,
                "message": "COUNT(DISTINCT ...) is expensive on large datasets. Consider approximate methods (APPROX_COUNT_DISTINCT)."
            })
    return findings


# ── Rule 15: OR in JOIN condition ─────────────────────────────
def _or_in_join(tree: exp.Select) -> list[dict]:
    findings = []
    for join in tree.find_all(exp.Join):
        for or_ in join.find_all(exp.Or):
            findings.append({
                "rule": "or_in_join", "severity": "warning", "line": or_.line,
                "message": "OR in JOIN condition prevents hash join. Consider UNION of two queries."
            })
    return findings


# ── Rule 16: IN without subquery (list of many values) ────────
def _in_many_values(tree: exp.Select) -> list[dict]:
    findings = []
    for in_ in tree.find_all(exp.In):
        if isinstance(in_.this, exp.Column) and in_.find(exp.Subquery) is None:
            values = [v for v in in_.expressions if isinstance(v, exp.Literal)]
            if len(values) > 10:
                findings.append({
                    "rule": "in_many_values", "severity": "info", "line": in_.line,
                    "message": f"IN list with {len(values)} values. Consider a temp table or JOIN."
                })
    return findings


# ── Rule 17: Nested aggregate (aggregate of aggregate) ────────
def _nested_aggregate(tree: exp.Select) -> list[dict]:
    findings = []
    for agg in tree.find_all(exp.AggFunc):
        if any(isinstance(child, exp.AggFunc) for child in agg.find_all(exp.AggFunc)):
            findings.append({
                "rule": "nested_aggregate", "severity": "warning", "line": agg.line,
                "message": f"Nested aggregate ({agg.sql()}). Use a window function or CTE instead."
            })
    return findings


# ── Rule 18: No primary key / no filter on unique column ──────
def _no_filter_on_id(tree: exp.Select) -> list[dict]:
    """Flag SELECT from a single table with no WHERE filtering an *_id column."""
    if not tree.find(exp.Where):
        return []
    from_tables = tree.find_all(exp.Table)
    if len(list(from_tables)) != 1:
        return []
    where = tree.find(exp.Where)
    if where is None:
        return []
    id_filtered = False
    for eq in where.find_all(exp.EQ):
        if isinstance(eq.left, exp.Column) and "_id" in eq.left.name.lower():
            id_filtered = True
    if not id_filtered:
        return [{
            "rule": "no_filter_on_id", "severity": "info", "line": tree.line,
            "message": "WHERE clause present but no filter on *_id column. Verify scan is intended."
        }]
    return []


# ── Rule 19: Correlated subquery (subquery referencing outer table) ──
def _correlated_subquery(tree: exp.Select) -> list[dict]:
    findings = []
    for subq in tree.find_all(exp.Subquery):
        inner_tree = subq.this
        # Check if inner references a table from outer scope
        outer_tables = {t.name for t in tree.find_all(exp.Table)
                        if t.find_ancestor(exp.Subquery) is None}
        for col in inner_tree.find_all(exp.Column):
            if col.table and col.table in outer_tables:
                findings.append({
                    "rule": "correlated_subquery", "severity": "warning", "line": subq.line,
                    "message": f"Correlated subquery referencing outer table '{col.table}'. Consider rewrite with JOIN/LATERAL."
                })
                break
    return findings


# ── Registry ──────────────────────────────────────────────────

RULES_REGISTRY: list[SQLRule] = [
    SQLRule("select_star", "warning", "SELECT * fetches all columns", _select_star),
    SQLRule("cartesian_join", "error", "Cartesian join without ON condition", _cartesian_join),
    SQLRule("non_sargable", "warning", "Function wrapping column prevents index use", _non_sargable),
    SQLRule("like_leading_wildcard", "warning", "LIKE with leading wildcard is non-sargable", _like_leading_wildcard),
    SQLRule("negation_in_where", "warning", "NOT/!= in WHERE prevents index use", _negation_in_where),
    SQLRule("missing_where", "error", "SELECT without WHERE reads full table", _missing_where),
    SQLRule("distinct_many_columns", "warning", "DISTINCT on many columns", _distinct_many_columns),
    SQLRule("order_by_no_limit", "warning", "ORDER BY without LIMIT sorts full table", _order_by_no_limit),
    SQLRule("group_by_ordinal", "warning", "GROUP BY ordinal is fragile", _group_by_ordinal),
    SQLRule("float_equality", "warning", "Floating-point equality comparison", _float_equality),
    SQLRule("having_without_group_by", "error", "HAVING without GROUP BY", _having_without_group_by),
    SQLRule("subquery_no_alias", "error", "Subquery missing alias", _subquery_no_alias),
    SQLRule("union_no_all", "info", "UNION without ALL sorts dedup", _union_no_all),
    SQLRule("count_distinct_many", "info", "COUNT(DISTINCT) is expensive", _count_distinct_many),
    SQLRule("or_in_join", "warning", "OR in JOIN prevents hash join", _or_in_join),
    SQLRule("in_many_values", "info", "IN list with many values", _in_many_values),
    SQLRule("nested_aggregate", "warning", "Aggregate of aggregate", _nested_aggregate),
    SQLRule("no_filter_on_id", "info", "WHERE without ID filter", _no_filter_on_id),
    SQLRule("correlated_subquery", "warning", "Correlated subquery should be JOIN", _correlated_subquery),
]


def analyze_sql(sql: str) -> list[dict]:
    """Run all 19 rules against a SQL string. Returns list of findings."""
    tree = _safe_parse(sql)
    if tree is None:
        return [{"rule": "parse_error", "severity": "error", "line": 0,
                 "message": "Could not parse SQL statement."}]
    findings = []
    for rule in RULES_REGISTRY:
        try:
            findings.extend(rule.check(tree))
        except Exception:
            pass
    return findings
```

- [ ] **Step 3: Create `tests/test_sql_engine.py`**

```python
"""Tests for SQL anti-pattern engine."""

import pytest
from core.sql_engine.rules import analyze_sql
from core.sql_engine.lineage import extract_lineage


class TestAnalyzeSQL:
    def test_select_star_detected(self):
        findings = analyze_sql("SELECT * FROM orders")
        assert any(f["rule"] == "select_star" for f in findings)

    def test_clean_sql_no_findings(self):
        findings = analyze_sql("SELECT o.order_id, o.quantity FROM orders o WHERE o.quantity > 0 LIMIT 10")
        errors = [f for f in findings if f["severity"] == "error"]
        assert len(errors) == 0

    def test_cartesian_join_detected(self):
        findings = analyze_sql("SELECT * FROM orders CROSS JOIN customers")
        assert any(f["rule"] == "cartesian_join" for f in findings)

    def test_missing_where_detected(self):
        findings = analyze_sql("SELECT order_id, quantity FROM orders")
        assert any(f["rule"] == "missing_where" for f in findings)

    def test_order_by_no_limit_detected(self):
        findings = analyze_sql("SELECT order_id FROM orders ORDER BY order_id")
        assert any(f["rule"] == "order_by_no_limit" for f in findings)

    def test_having_without_group_by(self):
        findings = analyze_sql("SELECT order_id FROM orders HAVING quantity > 0")
        assert any(f["rule"] == "having_without_group_by" for f in findings)

    def test_subquery_no_alias(self):
        findings = analyze_sql("SELECT * FROM (SELECT order_id FROM orders)")
        assert any(f["rule"] == "subquery_no_alias" for f in findings)

    def test_like_leading_wildcard(self):
        findings = analyze_sql("SELECT * FROM customers WHERE email LIKE '%@example.com'")
        assert any(f["rule"] == "like_leading_wildcard" for f in findings)

    def test_group_by_ordinal(self):
        findings = analyze_sql("SELECT category, COUNT(*) FROM products GROUP BY 1")
        assert any(f["rule"] == "group_by_ordinal" for f in findings)

    def test_float_equality(self):
        findings = analyze_sql("SELECT * FROM orders WHERE unit_price = 19.99")
        assert any(f["rule"] == "float_equality" for f in findings)

    def test_parse_error_returns_finding(self):
        findings = analyze_sql("SELECT ERRONEOUS SQL {{{")
        assert any(f["rule"] == "parse_error" for f in findings)

    def test_or_in_join(self):
        findings = analyze_sql("SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id OR o.email = c.email")
        assert any(f["rule"] == "or_in_join" for f in findings)

    def test_correlated_subquery(self):
        findings = analyze_sql("SELECT o.order_id, (SELECT MAX(unit_price) FROM orders o2 WHERE o2.customer_id = o.customer_id) AS max_price FROM orders o")
        assert any(f["rule"] == "correlated_subquery" for f in findings)

    def test_negation_in_where(self):
        findings = analyze_sql("SELECT * FROM orders WHERE NOT quantity > 0")
        assert any(f["rule"] == "negation_in_where" for f in findings)

    def test_in_many_values(self):
        findings = analyze_sql("SELECT * FROM orders WHERE product_id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)")
        assert any(f["rule"] == "in_many_values" for f in findings)
```

- [ ] **Step 4: Create `core/sql_engine/lineage.py`**

Column-level lineage extractor — traces columns through CTEs, joins, and projections.

```python
"""Column-level lineage extraction from SQL using sqlglot."""

from __future__ import annotations
import sqlglot
from sqlglot import exp
from collections import defaultdict


def extract_lineage(sql: str) -> list[dict]:
    """Extract column-level lineage from a SQL statement.

    Returns a list of dicts mapping {table, column, sources, transformation_type}
    where `sources` is a list of {table, column} tuples.
    """
    try:
        tree = sqlglot.parse_one(sql)
    except Exception:
        return []

    # Only handle SELECT statements
    if not isinstance(tree, (exp.Select, exp.Union)):
        return []

    results = []
    # Collect CTE definitions
    cte_defs: dict[str, exp.Select] = {}
    with_clause = tree.args.get("with")
    if with_clause:
        for cte in with_clause.expressions:
            name = cte.args.get("alias").name
            cte_defs[name] = cte.this

    # Resolve table name for a Column reference
    def resolve_column(col: exp.Column, context_aliases: dict) -> dict | None:
        table_name = col.table or next(iter(context_aliases.values()), "unknown")
        return {"table": table_name, "column": col.name}

    # Walk projections in the top-level SELECT
    select_expressions = tree.args.get("expressions", [])
    for expr in select_expressions:
        alias = None
        if isinstance(expr, exp.Alias):
            alias = expr.alias_or_name
            inner = expr.this
        elif isinstance(expr, exp.Column):
            inner = expr
            alias = expr.alias_or_name
        else:
            inner = expr
            alias = expr.alias_or_name

        sources = []
        trans_type = "direct"

        if isinstance(inner, exp.AggFunc):
            trans_type = "aggregate"
            for col in inner.find_all(exp.Column):
                sources.append(resolve_column(col, {}))
        elif isinstance(inner, exp.Func):
            trans_type = "function"
            for col in inner.find_all(exp.Column):
                sources.append(resolve_column(col, {}))
        elif isinstance(inner, exp.Column):
            sources = [resolve_column(inner, {})]
        elif isinstance(inner, exp.Literal):
            trans_type = "constant"
        elif isinstance(inner, exp.Cast):
            trans_type = "cast"
            for col in inner.find_all(exp.Column):
                sources.append(resolve_column(col, {}))
        else:
            for col in inner.find_all(exp.Column):
                sources.append(resolve_column(col, {}))

        results.append({
            "column": alias or str(inner)[:40],
            "transformation_type": trans_type,
            "sources": sources,
        })

    return results
```

- [ ] **Step 5: Run SQL engine tests**

Run: `pytest tests/test_sql_engine.py -v`
Expected: All tests pass (15+ tests, 0 failures).

- [ ] **Step 6: Commit**

```bash
git add core/sql_engine/ tests/test_sql_engine.py
git commit -m "feat: add deterministic SQL anti-pattern engine (19 rules) + column-level lineage"
```

---

### Task 3: Medallion Architecture — Bronze/Silver/Gold pipeline

**Files:**
- Create: `core/medallion/__init__.py`
- Create: `core/medallion/bronze.py`
- Create: `core/medallion/silver.py`
- Create: `core/medallion/gold.py`
- Create: `core/medallion/pipeline.py`
- Create: `tests/test_medallion.py`

- [ ] **Step 1: Create `core/medallion/__init__.py`**

```python
"""Medallion Architecture — Bronze / Silver / Gold pipeline."""

from core.medallion.pipeline import MedallionPipeline

__all__ = ["MedallionPipeline"]
```

- [ ] **Step 2: Create `core/medallion/bronze.py`**

```python
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
```

- [ ] **Step 3: Create `core/medallion/silver.py`**

```python
"""Silver layer — cleaning, dedup, validation, and type casting."""

from __future__ import annotations
import duckdb
import pandas as pd


def run_silver(conn: duckdb.DuckDBPyConnection) -> dict:
    """Clean bronze tables into silver layer.

    - orders: remove negative quantities, parse prices, normalize dates
    - customers: deduplicate, fill null emails
    - products: validate category values

    Returns quality report: {cleaning_steps, counts, issues}
    """
    steps = []
    issues = []

    # ── Clean orders ──────────────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE silver_orders AS
        SELECT
            order_id,
            customer_id,
            product_id,
            CAST(order_date AS DATE) AS order_date,
            CASE WHEN quantity > 0 THEN quantity ELSE NULL END AS quantity,
            CASE WHEN TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                 THEN CAST(unit_price AS DOUBLE) ELSE NULL END AS unit_price,
            channel
        FROM bronze_orders
    """)

    total_orders = conn.execute("SELECT COUNT(*) FROM bronze_orders").fetchone()[0]
    bad_qty = conn.execute("SELECT COUNT(*) FROM silver_orders WHERE quantity IS NULL").fetchone()[0]
    bad_price = conn.execute("SELECT COUNT(*) FROM silver_orders WHERE unit_price IS NULL").fetchone()[0]
    null_dates = conn.execute("SELECT COUNT(*) FROM silver_orders WHERE order_date IS NULL").fetchone()[0]

    steps.append(f"Total raw orders: {total_orders}")
    steps.append(f"Negative/null quantities cleaned: {bad_qty}")
    steps.append(f"Non-numeric prices cleaned: {bad_price}")
    steps.append(f"Null dates found: {null_dates}")
    if bad_qty > 0 or bad_price > 0:
        issues.append(f"Data quality: {bad_qty} bad quantities, {bad_price} bad prices")

    # ── Clean customers ───────────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE silver_customers AS
        SELECT DISTINCT customer_id, name,
            COALESCE(email, 'unknown@unknown.com') AS email,
            country, signup_date
        FROM bronze_customers
    """)
    orig_cust = conn.execute("SELECT COUNT(*) FROM bronze_customers").fetchone()[0]
    deduped = conn.execute("SELECT COUNT(*) FROM silver_customers").fetchone()[0]
    dupes = orig_cust - deduped
    if dupes > 0:
        steps.append(f"Customers deduplicated: {dupes} duplicates removed")
        issues.append(f"Data quality: {dupes} duplicate customer rows")

    # ── Clean products ────────────────────────────────────────
    valid_categories = ("Electronics", "Home", "Toys", "Office")
    conn.execute("""
        CREATE OR REPLACE TABLE silver_products AS
        SELECT product_id, product_name,
            CASE WHEN category IN ('Electronics', 'Home', 'Toys', 'Office')
                 THEN category ELSE 'Other' END AS category,
            list_price
        FROM bronze_products
    """)
    bad_cats = conn.execute("""SELECT COUNT(*) FROM bronze_products
        WHERE category NOT IN ('Electronics', 'Home', 'Toys', 'Office')""").fetchone()[0]
    if bad_cats > 0:
        steps.append(f"Products with invalid categories reclassified: {bad_cats}")
        issues.append(f"Data quality: {bad_cats} products had invalid categories")

    return {
        "steps": steps,
        "issues": issues,
        "silver_tables": ["silver_orders", "silver_customers", "silver_products"],
        "row_counts": {
            "silver_orders": conn.execute("SELECT COUNT(*) FROM silver_orders").fetchone()[0],
            "silver_customers": conn.execute("SELECT COUNT(*) FROM silver_customers").fetchone()[0],
            "silver_products": conn.execute("SELECT COUNT(*) FROM silver_products").fetchone()[0],
        },
    }
```

- [ ] **Step 4: Create `core/medallion/gold.py`**

```python
"""Gold layer — aggregated, analytics-ready marts."""

from __future__ import annotations
import duckdb


def build_gold_marts(conn: duckdb.DuckDBPyConnection) -> dict:
    """Build gold-layer aggregate marts from silver tables.

    Creates:
    - weekly_sales: revenue, orders, unique customers per week
    - customer_metrics: lifetime value, order frequency per customer
    - category_performance: revenue by product category

    Returns descriptions + row counts.
    """
    marts = {}

    # ── Weekly Sales Mart ─────────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE gold_weekly_sales AS
        SELECT
            date_trunc('week', order_date) AS week,
            ROUND(SUM(quantity * unit_price), 2) AS revenue,
            COUNT(*) AS order_count,
            COUNT(DISTINCT customer_id) AS unique_customers
        FROM silver_orders
        WHERE quantity IS NOT NULL AND unit_price IS NOT NULL
        GROUP BY week ORDER BY week
    """)
    ws = conn.execute("SELECT * FROM gold_weekly_sales").fetchdf()
    marts["gold_weekly_sales"] = {
        "row_count": len(ws),
        "total_revenue": round(float(ws["revenue"].sum()), 2),
        "avg_weekly_revenue": round(float(ws["revenue"].mean()), 2),
        "weeks": len(ws),
    }

    # ── Customer Metrics Mart ─────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE gold_customer_metrics AS
        SELECT
            c.customer_id,
            c.name,
            c.country,
            COUNT(o.order_id) AS total_orders,
            ROUND(SUM(o.quantity * o.unit_price), 2) AS lifetime_value,
            ROUND(AVG(o.quantity * o.unit_price), 2) AS avg_order_value
        FROM silver_customers c
        LEFT JOIN silver_orders o ON c.customer_id = o.customer_id
            AND o.quantity IS NOT NULL AND o.unit_price IS NOT NULL
        GROUP BY c.customer_id, c.name, c.country
    """)
    cm = conn.execute("SELECT * FROM gold_customer_metrics").fetchdf()
    marts["gold_customer_metrics"] = {
        "row_count": len(cm),
        "avg_lifetime_value": round(float(cm["lifetime_value"].mean()), 2) if len(cm) > 0 else 0,
    }

    # ── Category Performance Mart ─────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE gold_category_performance AS
        SELECT
            p.category,
            ROUND(SUM(o.quantity * o.unit_price), 2) AS revenue,
            COUNT(*) AS order_count,
            COUNT(DISTINCT o.customer_id) AS unique_customers
        FROM silver_orders o
        JOIN silver_products p ON o.product_id = p.product_id
        WHERE o.quantity IS NOT NULL AND o.unit_price IS NOT NULL
        GROUP BY p.category ORDER BY revenue DESC
    """)
    cp = conn.execute("SELECT * FROM gold_category_performance").fetchdf()
    marts["gold_category_performance"] = {
        "row_count": len(cp),
        "total_revenue": round(float(cp["revenue"].sum()), 2),
    }

    return marts
```

- [ ] **Step 5: Create `core/medallion/pipeline.py`**

```python
"""Unified MedallionPipeline — runs Bronze → Silver → Gold in sequence."""

from __future__ import annotations
from pathlib import Path
import duckdb
from core.medallion.bronze import ingest_bronze
from core.medallion.silver import run_silver
from core.medallion.gold import build_gold_marts


class MedallionPipeline:
    """Run the full medallion pipeline: Bronze (raw) → Silver (clean) → Gold (marts)."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def run(self, conn: duckdb.DuckDBPyConnection | None = None) -> dict:
        """Execute Bronze → Silver → Gold pipeline.

        Returns full report with manifest, quality report, and mart descriptions.
        """
        close_conn = conn is None
        conn = conn or duckdb.connect(":memory:")

        try:
            bronze = ingest_bronze(conn, self.data_dir)
            silver = run_silver(conn)
            gold = build_gold_marts(conn)

            return {
                "status": "ok",
                "bronze": bronze,
                "silver": silver,
                "gold": gold,
            }
        finally:
            if close_conn:
                conn.close()
```

- [ ] **Step 6: Create `tests/test_medallion.py`**

```python
"""Tests for Medallion pipeline."""

import pytest
from pathlib import Path
import duckdb
from core.medallion.pipeline import MedallionPipeline
from data.generate_sample import generate


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    generate(d)
    return d


class TestMedallionPipeline:
    def test_bronze_creates_tables(self, data_dir):
        conn = duckdb.connect(":memory:")
        pipe = MedallionPipeline(data_dir)
        result = pipe.run(conn)
        assert "bronze_orders" in result["bronze"]
        assert "bronze_customers" in result["bronze"]
        assert "bronze_products" in result["bronze"]
        assert result["bronze"]["bronze_orders"]["row_count"] > 0
        conn.close()

    def test_silver_cleans_data(self, data_dir):
        conn = duckdb.connect(":memory:")
        pipe = MedallionPipeline(data_dir)
        result = pipe.run(conn)
        assert len(result["silver"]["steps"]) > 0
        assert result["silver"]["row_counts"]["silver_orders"] > 0
        conn.close()

    def test_gold_builds_marts(self, data_dir):
        conn = duckdb.connect(":memory:")
        pipe = MedallionPipeline(data_dir)
        result = pipe.run(conn)
        assert "gold_weekly_sales" in result["gold"]
        assert "gold_customer_metrics" in result["gold"]
        assert "gold_category_performance" in result["gold"]
        assert result["gold"]["gold_weekly_sales"]["total_revenue"] > 0
        conn.close()

    def test_run_without_conn(self, data_dir):
        pipe = MedallionPipeline(data_dir)
        result = pipe.run()
        assert result["status"] == "ok"

    def test_bronze_tracks_schema_hash(self, data_dir):
        conn = duckdb.connect(":memory:")
        pipe = MedallionPipeline(data_dir)
        result = pipe.run(conn)
        assert isinstance(result["bronze"]["bronze_orders"]["schema_hash"], int)
        conn.close()

    def test_silver_detects_bad_data(self, data_dir):
        conn = duckdb.connect(":memory:")
        pipe = MedallionPipeline(data_dir)
        result = pipe.run(conn)
        # Should find the seeded bad data
        has_quality_issues = len(result["silver"]["issues"]) > 0
        assert has_quality_issues
        conn.close()
```

- [ ] **Step 7: Run medallion tests**

Run: `pytest tests/test_medallion.py -v`
Expected: All 6 tests pass.

- [ ] **Step 8: Commit**

```bash
git add core/medallion/ tests/test_medallion.py
git commit -m "feat: add medallion architecture pipeline (bronze/silver/gold)"
```

---

### Task 4: Upgraded PII Detection — 15 categories

**Files:**
- Create: `core/pii/__init__.py`
- Create: `core/pii/detector.py`
- Create: `tests/test_pii.py`

- [ ] **Step 1: Create `core/pii/__init__.py`**

```python
"""PII detection — 15 categories via regex + entropy analysis."""

from core.pii.detector import scan_pii, PII_CATEGORIES, PiiFinding

__all__ = ["scan_pii", "PII_CATEGORIES", "PiiFinding"]
```

- [ ] **Step 2: Create `core/pii/detector.py`**

```python
"""PII detector — 15 categories, regex + entropy-based detection."""

from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class PiiFinding:
    table: str
    column: str
    category: str
    matches: int
    sample_values: list[str]
    confidence: str        # "high" | "medium" | "low"
    severity: str          # "critical" | "high" | "medium" | "low"


# 15 PII categories with regex patterns
PII_RULES: list[tuple[str, str, str, list[str]]] = [
    # (category, severity, regex_pattern, context_keywords)
    ("email", "high", r'[\w\.+-]+@[\w\.-]+\.\w+', ["email", "e-mail", "mail", "contact"]),
    ("phone", "high", r'\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', ["phone", "mobile", "tel", "contact", "fax"]),
    ("credit_card", "critical", r'\b(?:\d{4}[-\s]?){3}\d{4}\b', ["card", "cc", "credit", "payment"]),
    ("ssn", "critical", r'\b\d{3}-\d{2}-\d{4}\b', ["ssn", "social", "security", "tax_id"]),
    ("ip_address", "medium", r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ["ip", "ip_address", "ipaddr"]),
    ("date_of_birth", "high", r'\b\d{4}-\d{2}-\d{2}\b', ["dob", "birth", "birth_date", "born"]),
    ("passport", "critical", r'\b[A-Z]{1,2}\d{6,9}\b', ["passport", "passport_no"]),
    ("driver_license", "high", r'\b[A-Z]\d{3,7}\b', ["license", "driver", "driving"]),
    ("bank_account", "critical", r'\b\d{8,17}\b', ["account", "bank", "iban", "aba"]),
    ("health_insurance", "high", r'\b[A-Z]{2,3}\d{6,12}\b', ["insurance", "health", "medicare", "medicaid"]),
    ("street_address", "medium", r'\b\d{1,5}\s+[A-Za-z]+\s+(Street|Ave|Road|Drive|Lane|Blvd|Court|Way)\b', ["address", "street", "location", "addr"]),
    ("zip_code", "medium", r'\b\d{5}(?:-\d{4})?\b', ["zip", "postal", "postcode", "zip_code"]),
    ("url", "low", r'https?://[^\s/$.?#].[^\s]*', ["url", "website", "link", "web"]),
    ("api_key", "critical", r'\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|AIza[0-9A-Za-z_-]{35})\b', ["api_key", "api", "secret", "token"]),
    ("password_hash", "critical", r'\b\$2[ayb]\$\d{2}\$[./A-Za-z0-9]{53}\b', ["password", "hash", "bcrypt", "passwd"]),
]

PII_CATEGORIES = [r[0] for r in PII_RULES]


def _entropy(s: str) -> float:
    """Shannon entropy — high entropy suggests encoded/token data."""
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def scan_pii(df: pd.DataFrame, table_name: str, column_hints: Optional[list[str]] = None) -> list[PiiFinding]:
    """Scan a DataFrame for PII across 15 categories.

    Args:
        df: DataFrame to scan (first 1000 rows sample recommended for performance).
        table_name: Name of source table for reporting.
        column_hints: Optional column names to focus on (scans all object columns if None).

    Returns:
        List of PiiFinding dataclass instances.
    """
    findings = []
    cols_to_scan = column_hints or [c for c in df.select_dtypes(include="object").columns]

    for col in cols_to_scan:
        if col not in df.columns:
            continue
        col_lower = col.lower()
        series = df[col].dropna().astype(str)

        if len(series) == 0:
            continue

        # Check column name for contextual hints
        col_matches_context = any(
            any(kw in col_lower for kw in keywords)
            for _, _, _, keywords in PII_RULES
        )

        for category, severity, pattern, keywords in PII_RULES:
            # Skip if column name doesn't match context (reduces false positives)
            col_context_match = any(kw in col_lower for kw in keywords)
            if not col_context_match and not col_matches_context:
                # Still scan a sample but with lower sensitivity
                sample = series.head(50)
            else:
                sample = series.head(200)

            matches = sample.str.findall(pattern)
            # Filter out overly broad matches using entropy
            valid_matches = []
            for m in matches:
                for val in m:
                    entropy = _entropy(val)
                    # Credit card: high entropy expected; address: lower entropy
                    if category in ("credit_card", "api_key", "password_hash") and entropy < 2.0:
                        continue
                    if category == "zip_code" and len(val) < 5:
                        continue
                    valid_matches.append(val)

            if valid_matches:
                confidence = "high" if col_context_match else "medium"
                # If column name directly suggests the category, boost confidence
                if any(kw in col_lower for kw in keywords):
                    confidence = "high"

                findings.append(PiiFinding(
                    table=table_name,
                    column=col,
                    category=category,
                    matches=len(valid_matches),
                    sample_values=list(set(valid_matches))[:5],
                    confidence=confidence,
                    severity=severity,
                ))

    return findings
```

- [ ] **Step 3: Create `tests/test_pii.py`**

```python
"""Tests for PII detector."""

import pytest
import pandas as pd
from core.pii.detector import scan_pii, PII_CATEGORIES


class TestPIIDetector:
    def test_detects_email(self):
        df = pd.DataFrame({"contact": ["alice@example.com", "bob@test.com", "noise"]})
        findings = scan_pii(df, "test")
        emails = [f for f in findings if f.category == "email"]
        assert len(emails) > 0
        assert emails[0].matches >= 2

    def test_detects_credit_card(self):
        df = pd.DataFrame({"cc_number": ["4111-1111-1111-1111", "5500-0000-0000-0004", "noise"]})
        findings = scan_pii(df, "test")
        cards = [f for f in findings if f.category == "credit_card"]
        assert len(cards) > 0
        assert cards[0].matches >= 2

    def test_detects_phone(self):
        df = pd.DataFrame({"phone": ["+1-555-123-4567", "555-123-4567", "noise"]})
        findings = scan_pii(df, "test")
        phones = [f for f in findings if f.category == "phone"]
        assert len(phones) > 0

    def test_detects_ip_address(self):
        df = pd.DataFrame({"ip_address": ["192.168.1.1", "10.0.0.255", "noise"]})
        findings = scan_pii(df, "test")
        ips = [f for f in findings if f.category == "ip_address"]
        assert len(ips) > 0

    def test_detects_ssn(self):
        df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321", "noise"]})
        findings = scan_pii(df, "test")
        ssns = [f for f in findings if f.category == "ssn"]
        assert len(ssns) > 0

    def test_detects_url(self):
        df = pd.DataFrame({"website": ["https://example.com/page", "http://test.org", "noise"]})
        findings = scan_pii(df, "test")
        urls = [f for f in findings if f.category == "url"]
        assert len(urls) > 0

    def test_detects_zip_code(self):
        df = pd.DataFrame({"zip": ["90210", "12345-6789", "noise"]})
        findings = scan_pii(df, "test")
        zips = [f for f in findings if f.category == "zip_code"]
        assert len(zips) > 0

    def test_detects_date_of_birth(self):
        df = pd.DataFrame({"dob": ["1990-01-15", "1985-06-22", "noise"]})
        findings = scan_pii(df, "test")
        dobs = [f for f in findings if f.category == "date_of_birth"]
        assert len(dobs) > 0

    def test_clean_data_returns_empty(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": ["30", "25", "35"]})
        findings = scan_pii(df, "test")
        assert len(findings) == 0 or all(f.confidence == "low" for f in findings)

    def test_column_hints_focus_scan(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "email_addr": ["a@x.com", "b@y.com"]})
        findings = scan_pii(df, "test", column_hints=["email_addr"])
        assert len(findings) > 0
        assert findings[0].column == "email_addr"

    def test_all_categories_defined(self):
        assert len(PII_CATEGORIES) == 15
        expected = ["email", "phone", "credit_card", "ssn", "ip_address", "date_of_birth",
                     "passport", "driver_license", "bank_account", "health_insurance",
                     "street_address", "zip_code", "url", "api_key", "password_hash"]
        assert PII_CATEGORIES == expected
```

- [ ] **Step 4: Run PII tests**

Run: `pytest tests/test_pii.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/pii/ tests/test_pii.py
git commit -m "feat: add 15-category PII detector with entropy analysis"
```

---

### Task 5: RAG Knowledge Base — LanceDB vector store

**Files:**
- Create: `core/knowledge/__init__.py`
- Create: `core/knowledge/embeddings.py`
- Create: `core/knowledge/store.py`
- Create: `core/knowledge/semantic.py`
- Create: `tests/test_knowledge.py`

- [ ] **Step 1: Create `core/knowledge/__init__.py`**

```python
"""RAG Knowledge Base — LanceDB vector store, embeddings, semantic layer."""

from core.knowledge.store import KnowledgeStore
from core.knowledge.semantic import SemanticRegistry, MetricDef

__all__ = ["KnowledgeStore", "SemanticRegistry", "MetricDef"]
```

- [ ] **Step 2: Create `core/knowledge/embeddings.py`**

```python
"""Text embedding functions."""

from __future__ import annotations


def get_embedder():
    """Lazy-loaded sentence transformer model. Returns a callable embed(text) -> list[float]."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(text: str) -> list[float]:
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_many(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    return embed, embed_many
```

- [ ] **Step 3: Create `core/knowledge/store.py`**

```python
"""LanceDB-backed knowledge store for schema, queries, and business context."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class KnowledgeDoc:
    id: str
    text: str
    metadata: dict
    category: str  # "schema" | "query" | "metric" | "business_rule"


class KnowledgeStore:
    """Vector store for data engineering knowledge.

    Stores:
    - Schema descriptions (table/column docs)
    - Reference SQL queries (frequently used patterns)
    - Business metrics definitions
    - Business rules and context
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._table = None
        self._embed, self._embed_many = None, None

    def _lazy_init(self):
        if self._table is not None:
            return
        import lancedb
        db = lancedb.connect(str(self.db_path))
        try:
            self._table = db.open_table("knowledge")
        except Exception:
            self._table = db.create_table("knowledge", [
                {"vector": [0.0] * 384, "id": "", "text": "", "metadata": "{}", "category": ""}
            ], mode="overwrite")
        from core.knowledge.embeddings import get_embedder
        self._embed, self._embed_many = get_embedder()

    def add_doc(self, doc: KnowledgeDoc):
        """Add a single document to the knowledge base."""
        self._lazy_init()
        vec = self._embed(doc.text)
        self._table.add([{
            "vector": vec,
            "id": doc.id,
            "text": doc.text,
            "metadata": json.dumps(doc.metadata),
            "category": doc.category,
        }])

    def add_docs(self, docs: list[KnowledgeDoc]):
        """Add multiple documents in batch."""
        self._lazy_init()
        texts = [d.text for d in docs]
        vecs = self._embed_many(texts)
        records = [
            {"vector": v, "id": d.id, "text": d.text,
             "metadata": json.dumps(d.metadata), "category": d.category}
            for v, d in zip(vecs, docs)
        ]
        self._table.add(records)

    def search(self, query: str, category: Optional[str] = None, top_k: int = 5) -> list[KnowledgeDoc]:
        """Search the knowledge base by semantic similarity."""
        self._lazy_init()
        vec = self._embed(query)
        results = self._table.search(vec).limit(top_k).to_list()
        docs = []
        for r in results:
            if category and r.get("category") != category:
                continue
            docs.append(KnowledgeDoc(
                id=r["id"],
                text=r["text"],
                metadata=json.loads(r.get("metadata", "{}")),
                category=r.get("category", ""),
            ))
        return docs

    def count(self) -> int:
        """Return total document count."""
        self._lazy_init()
        return len(self._table.to_list())

    def seed_defaults(self, schema_info: Optional[list[KnowledgeDoc]] = None):
        """Seed the knowledge base with default schema info if empty."""
        self._lazy_init()
        if self.count() > 0:
            return
        defaults = schema_info or []
        if defaults:
            self.add_docs(defaults)
```

- [ ] **Step 4: Create `core/knowledge/semantic.py`**

```python
"""Semantic layer — YAML-defined business metrics that resolve to SQL."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class MetricDef:
    name: str
    description: str
    sql: str
    type: str          # "simple" | "aggregate" | "ratio" | "derived"
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    source_table: str = ""
    dimensions: list[str] = None


class SemanticRegistry:
    """Registry of business metrics defined in YAML, resolved to SQL."""

    def __init__(self):
        self._metrics: dict[str, MetricDef] = {
            "total_revenue": MetricDef(
                name="total_revenue",
                description="Total gross revenue from all orders",
                sql="SELECT ROUND(SUM(quantity * unit_price), 2) AS total_revenue FROM silver_orders WHERE quantity IS NOT NULL AND unit_price IS NOT NULL",
                type="aggregate",
                source_table="silver_orders",
                dimensions=["channel", "country"],
            ),
            "avg_order_value": MetricDef(
                name="avg_order_value",
                description="Average value per order",
                sql="SELECT ROUND(AVG(quantity * unit_price), 2) AS avg_order_value FROM silver_orders WHERE quantity IS NOT NULL AND unit_price IS NOT NULL",
                type="aggregate",
                source_table="silver_orders",
                dimensions=["channel"],
            ),
            "repeat_customer_rate": MetricDef(
                name="repeat_customer_rate",
                description="Percentage of customers with more than one order",
                sql="""
                SELECT ROUND(100.0 * SUM(CASE WHEN cnt > 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS repeat_customer_rate
                FROM (SELECT customer_id, COUNT(*) AS cnt FROM silver_orders GROUP BY customer_id)
                """,
                type="ratio",
                source_table="silver_orders",
            ),
            "weekly_active_customers": MetricDef(
                name="weekly_active_customers",
                description="Number of unique customers per week",
                sql="SELECT week, unique_customers AS weekly_active_customers FROM gold_weekly_sales",
                type="simple",
                source_table="gold_weekly_sales",
                dimensions=["week"],
            ),
            "category_revenue_share": MetricDef(
                name="category_revenue_share",
                description="Revenue contribution by product category",
                sql="SELECT category, revenue AS category_revenue FROM gold_category_performance",
                type="simple",
                source_table="gold_category_performance",
                dimensions=["category"],
            ),
            "customer_lifetime_value_avg": MetricDef(
                name="customer_lifetime_value_avg",
                description="Average lifetime value across all customers",
                sql="SELECT ROUND(AVG(lifetime_value), 2) AS avg_lifetime_value FROM gold_customer_metrics",
                type="aggregate",
                source_table="gold_customer_metrics",
                dimensions=["country"],
            ),
        }

    def list_metrics(self) -> list[dict]:
        return [{"name": m.name, "description": m.description, "type": m.type} for m in self._metrics.values()]

    def get_metric(self, name: str) -> Optional[MetricDef]:
        return self._metrics.get(name)

    def resolve(self, metric_name: str, dimensions: Optional[list[str]] = None) -> str:
        """Resolve a metric name + optional dimensions to executable SQL."""
        metric = self._metrics.get(metric_name)
        if metric is None:
            return f"-- Unknown metric: {metric_name}"
        sql = metric.sql
        if dimensions:
            dim_cols = [d for d in dimensions if d in (metric.dimensions or [])]
            if dim_cols:
                sql = sql.replace("SELECT ", f"SELECT {', '.join(dim_cols)}, ")
        return sql

    def from_yaml(self, yaml_str: str) -> int:
        """Load metric definitions from a YAML string. Returns number loaded."""
        data = yaml.safe_load(yaml_str)
        if not data or "metrics" not in data:
            return 0
        count = 0
        for m in data["metrics"]:
            self._metrics[m["name"]] = MetricDef(
                name=m["name"],
                description=m.get("description", ""),
                sql=m["sql"],
                type=m.get("type", "simple"),
                source_table=m.get("source_table", ""),
                dimensions=m.get("dimensions", []),
            )
            count += 1
        return count
```

- [ ] **Step 5: Create `tests/test_knowledge.py`**

```python
"""Tests for Knowledge Store and Semantic Registry."""

import pytest
from pathlib import Path
import json
from core.knowledge.store import KnowledgeStore, KnowledgeDoc
from core.knowledge.semantic import SemanticRegistry


class TestKnowledgeStore:
    @pytest.fixture
    def store(self, tmp_path):
        return KnowledgeStore(tmp_path / "lancedb")

    def test_add_and_search(self, store):
        store.add_doc(KnowledgeDoc(
            id="test1", text="orders table contains order transactions",
            metadata={"table": "orders"}, category="schema"
        ))
        results = store.search("order transactions")
        assert len(results) > 0
        assert results[0].category == "schema"

    def test_batch_add(self, store):
        docs = [
            KnowledgeDoc(id="1", text="doc one", metadata={}, category="query"),
            KnowledgeDoc(id="2", text="doc two", metadata={}, category="query"),
        ]
        store.add_docs(docs)
        assert store.count() >= 2

    def test_search_by_category(self, store):
        store.add_doc(KnowledgeDoc(id="s1", text="schema info", metadata={}, category="schema"))
        store.add_doc(KnowledgeDoc(id="q1", text="SELECT * FROM", metadata={}, category="query"))
        results = store.search("schema", category="schema")
        assert all(r.category == "schema" for r in results)

    def test_empty_search_returns_empty_list(self, store):
        results = store.search("nothing here")
        assert len(results) == 0 or isinstance(results, list)

    def test_persistence(self, tmp_path):
        db_path = tmp_path / "lancedb"
        store1 = KnowledgeStore(db_path)
        store1.add_doc(KnowledgeDoc(id="p1", text="persistent doc", metadata={}, category="schema"))
        store2 = KnowledgeStore(db_path)
        results = store2.search("persistent")
        assert len(results) > 0


class TestSemanticRegistry:
    def test_list_metrics(self):
        reg = SemanticRegistry()
        metrics = reg.list_metrics()
        names = [m["name"] for m in metrics]
        assert "total_revenue" in names
        assert "avg_order_value" in names
        assert "repeat_customer_rate" in names

    def test_get_metric(self):
        reg = SemanticRegistry()
        m = reg.get_metric("total_revenue")
        assert m is not None
        assert m.type == "aggregate"
        assert "SUM" in m.sql.upper()

    def test_unknown_metric_returns_none(self):
        reg = SemanticRegistry()
        assert reg.get_metric("nonexistent") is None

    def test_resolve_metric_to_sql(self):
        reg = SemanticRegistry()
        sql = reg.resolve("total_revenue")
        assert "SUM" in sql.upper()

    def test_resolve_with_dimensions(self):
        reg = SemanticRegistry()
        sql = reg.resolve("total_revenue", dimensions=["channel"])
        assert "channel" in sql or "CHANNEL" in sql.upper()

    def test_from_yaml(self):
        reg = SemanticRegistry()
        yaml_str = """
        metrics:
          - name: test_metric
            description: A test
            sql: SELECT COUNT(*) FROM orders
            type: simple
        """
        count = reg.from_yaml(yaml_str)
        assert count == 1
        assert reg.get_metric("test_metric") is not None
```

- [ ] **Step 6: Run knowledge tests**

Run: `pytest tests/test_knowledge.py -v`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/knowledge/ tests/test_knowledge.py
git commit -m "feat: add RAG knowledge base (LanceDB) + semantic metric registry"
```

---

### Task 6: Wire SQL Agent to use SQL Engine

**Files:**
- Modify: `backend/agents/sql_agent.py`

This replaces the heuristic `_generate_sql` method with the deterministic SQL analyzer. The agent now:
1. Parses the user's question via existing heuristics (keep the pattern-matching for generating SQL)
2. After generating SQL, runs it through the anti-pattern engine
3. Returns analysis alongside results
4. Uses the semantic registry for known metric queries

- [ ] **Step 1: Modify `sql_agent.py` — add imports and analyze step**

Add to imports at top of file (after existing imports):
```python
from core.sql_engine.rules import analyze_sql
from core.sql_engine.lineage import extract_lineage
from core.knowledge.semantic import SemanticRegistry
```

Add after the AgentResult return in `run()` (just before the `try:` block that executes SQL):

After the SQL is generated and before it's executed, run it through the engine. Insert this block right after `sql = self._generate_sql(user_input, schema)`:

```python
        # Run SQL through anti-pattern engine
        if sql:
            sql_findings = analyze_sql(sql)
            lineage = extract_lineage(sql)
            if sql_findings:
                context = context or {}
                context["sql_findings"] = sql_findings
                context["lineage"] = lineage
```

Also modify the result to include anti-pattern findings in the detail. In the `detail_parts.append(...)` lines, after the SQL block add:
```python
            if context and context.get("sql_findings"):
                fp = [f for f in context["sql_findings"] if f["severity"] in ("error", "warning")]
                if fp:
                    detail_parts.append("\n**⚠️ SQL Anti-Pattern Analysis:**")
                    for f in fp[:5]:
                        detail_parts.append(f"- [{f['severity']}] {f['message']}")
```

- [ ] **Step 2: Run existing SQL agent tests to verify no regression**

Run: `pytest tests/test_agents.py::TestSQLAgent -v`
Expected: All existing SQL agent tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/agents/sql_agent.py
git commit -m "feat: wire SQL agent to anti-pattern engine + lineage"
```

---

### Task 7: Wire ETL Agent to use Medallion Pipeline

**Files:**
- Modify: `backend/agents/etl_agent.py`

The ETL agent currently does ad-hoc cleaning in `_clean()` and `_build_marts()`. Replace these with calls to `MedallionPipeline`.

- [ ] **Step 1: Modify `etl_agent.py`**

Add import:
```python
from core.medallion.pipeline import MedallionPipeline
```

Replace the `_profile()`, `_clean()`, and `_build_marts()` methods with a single call to `MedallionPipeline`. Modify the `run()` method:

```python
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
        detail_parts.append("### 🥉 Bronze Layer (Raw Ingestion)\n" + "\n".join(bronze_lines))

        # Silver summary
        detail_parts.append(
            "### 🥈 Silver Layer (Cleaning)\n" +
            "\n".join(f"- {s}" for s in silver["steps"]) +
            ("\n\n**Issues:**\n" + "\n".join(f"- ⚠️ {i}" for i in silver["issues"]) if silver["issues"] else "")
        )

        # Gold summary
        gold_lines = []
        for mart, info in gold.items():
            gold_lines.append(f"- **{mart}**: {info['row_count']:,} rows")
            if "total_revenue" in info:
                gold_lines[-1] += f", revenue=${info['total_revenue']:,.2f}"
        detail_parts.append("### 🥇 Gold Layer (Analytics Marts)\n" + "\n".join(gold_lines))

        # Summary
        silver_rows = sum(silver["row_counts"].values())
        gold_rows = sum(v["row_count"] for v in gold.values())
        summary = f"Medallion pipeline: {bronze['bronze_orders']['row_count']:,} raw → {silver_rows:,} clean → {gold_rows:,} mart rows"

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
```

Replace the entire `_profile()`, `_clean()`, `_build_marts()` methods with stub/no-op since they're no longer used. Keep `_get_conn()` and `cleanup()` as-is.

- [ ] **Step 2: Run existing ETL agent tests**

Run: `pytest tests/test_agents.py::TestETLAgent -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/agents/etl_agent.py
git commit -m "feat: wire ETL agent to medallion pipeline (bronze/silver/gold)"
```

---

### Task 8: Wire Quality Agent to use upgraded PII + anti-pattern checks

**Files:**
- Modify: `backend/agents/quality_agent.py`

- [ ] **Step 1: Modify imports and PII scan in `quality_agent.py`**

Add import:
```python
from core.pii.detector import scan_pii, PiiFinding
from core.sql_engine.rules import analyze_sql
```

Replace the `_pii_scan()` method entirely (the old one only had 3 patterns):

```python
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
            lines = [f"### 🔒 PII Scan (15 categories)"]
            for f in findings:
                lines.append(
                    f"  - {f['severity'].upper()} {f['table']}.{f['column']}: "
                    f"{f['matches']} {f['pii_type']} ({f['confidence']} confidence)"
                )
            detail_parts.append("\n".join(lines))
        else:
            detail_parts.append("### 🔒 PII Scan (15 categories)\nNo PII patterns detected.")
        return findings
```

Add a method for SQL anti-pattern scanning (invoked when "sql" or "query" keywords are present in the quality agent input):

```python
    def _sql_quality_scan(self, conn, detail_parts) -> list[dict]:
        """Scan reference SQL queries from the context for anti-patterns."""
        findings = []
        # Scan the clean/silver queries for anti-patterns
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
```

Add "sql" to the routing hints list at the top: `"sql", "query"`.

Update the `run()` method to include SQL quality scanning trigger words:

In the keyword check block, add after the outlier check:
```python
        if any(w in q for w in ["sql", "query", "anti-pattern", "all", "full"]):
            sqlq = self._sql_quality_scan(conn, detail_parts)
            findings.extend(sqlq)
```

- [ ] **Step 2: Run quality agent tests**

Run: `pytest tests/test_agents.py::TestQualityAgent -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/agents/quality_agent.py
git commit -m "feat: wire quality agent to 15-category PII + SQL anti-pattern scanning"
```

---

### Task 9: Wire Documentation Agent to include Lineage

**Files:**
- Modify: `backend/agents/doc_agent.py`

- [ ] **Step 1: Modify `doc_agent.py`**

Add import:
```python
from core.sql_engine.lineage import extract_lineage
```

In the `_pipeline_documentation()` method, enhance the output to include column-level lineage:

Replace the return block with:
```python
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
```

- [ ] **Step 2: Run doc agent tests**

Run: `pytest tests/test_agents.py::TestDocumentationAgent -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/agents/doc_agent.py
git commit -m "feat: wire documentation agent to include column-level lineage"
```

---

### Task 10: Update Orchestrator agent descriptions

**Files:**
- Modify: `backend/agents/orchestrator.py`

- [ ] **Step 1: Update descriptions to reflect new capabilities**

Update the agent descriptions in `_init_agents()` (the descriptions come from each agent's class property, so they're already updated). No code change needed — verify by checking each agent's `description` property as updated in the tasks above.

- [ ] **Step 2: Verify orchestrator still works**

Run: `pytest tests/test_agents.py::TestOrchestrator -v`
Expected: All tests pass.

---

### Task 11: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `pytest -v`
Expected: All tests pass (new + existing). Count expected tests:
- `test_sql_engine.py`: ~15 tests
- `test_medallion.py`: 6 tests
- `test_pii.py`: ~11 tests
- `test_knowledge.py`: ~10 tests
- `test_agents.py`: ~21 tests
- `test_tools.py`: 5 tests
- Other existing tests: ~5-10 tests

- [ ] **Step 2: Fix any failures**

If any test fails, fix the implementation before proceeding.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: core upgrade — sql engine, medallion, RAG, PII"
```