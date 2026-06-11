"""Column-level lineage extraction from SQL using sqlglot."""

from __future__ import annotations
import sqlglot
from sqlglot import exp
from collections import defaultdict


def extract_lineage(sql: str) -> list[dict]:
    """Extract column-level lineage from a SQL statement.

    Returns a list of dicts mapping {column, transformation_type, sources}
    where `sources` is a list of {table, column} tuples.
    """
    try:
        tree = sqlglot.parse_one(sql)
    except Exception:
        return []

    if not isinstance(tree, (exp.Select, exp.Union)):
        return []

    results = []

    def resolve_column(col: exp.Column) -> dict:
        return {"table": col.table or "unknown", "column": col.name}

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
                sources.append(resolve_column(col))
        elif isinstance(inner, exp.Func):
            trans_type = "function"
            for col in inner.find_all(exp.Column):
                sources.append(resolve_column(col))
        elif isinstance(inner, exp.Column):
            sources = [resolve_column(inner)]
        elif isinstance(inner, exp.Literal):
            trans_type = "constant"
        elif isinstance(inner, exp.Cast):
            trans_type = "cast"
            for col in inner.find_all(exp.Column):
                sources.append(resolve_column(col))
        else:
            for col in inner.find_all(exp.Column):
                sources.append(resolve_column(col))

        results.append({
            "column": alias or str(inner)[:40],
            "transformation_type": trans_type,
            "sources": sources,
        })

    return results