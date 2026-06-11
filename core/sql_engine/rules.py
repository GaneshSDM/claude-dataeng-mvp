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
            "rule": "select_star", "severity": "warning", "line": 0,
            "message": "SELECT * fetches all columns. Name columns explicitly for clarity and stability."
        })
    return findings


# ── Rule 2: Cartesian join (CROSS JOIN or implicit) ───────────
def _cartesian_join(tree: exp.Select) -> list[dict]:
    findings = []
    for join in tree.find_all(exp.Join):
        if join.kind == "CROSS" or join.this is None:
            findings.append({
                "rule": "cartesian_join", "severity": "error", "line": 0,
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
                "rule": "non_sargable", "severity": "warning", "line": 0,
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
                "rule": "like_leading_wildcard", "severity": "warning", "line": 0,
                "message": "LIKE with leading '%' is non-sargable. Consider full-text search."
            })
    return findings


# ── Rule 5: Negation in WHERE (NOT, !=, <>) ───────────────────
def _negation_in_where(tree: exp.Select) -> list[dict]:
    findings = []
    for not_ in tree.find_all(exp.Not):
        if not_.find_ancestor(exp.Where):
            findings.append({
                "rule": "negation_in_where", "severity": "warning", "line": 0,
                "message": "Negation (NOT) in WHERE can prevent index use. Consider rewriting positively."
            })
    for neq in tree.find_all(exp.NEQ):
        if neq.find_ancestor(exp.Where):
            findings.append({
                "rule": "negation_in_where", "severity": "info", "line": 0,
                "message": "Inequality (!= / <>) comparison in WHERE. Typically non-sargable."
            })
    return findings


# ── Rule 6: Missing WHERE clause ─────────────────────────────
def _missing_where(tree: exp.Select) -> list[dict]:
    if not tree.find(exp.Where) and tree.find(exp.From):
        return [{
            "rule": "missing_where", "severity": "error", "line": 0,
            "message": "SELECT without WHERE clause — reads full table. Add WHERE or LIMIT."
        }]
    return []


# ── Rule 7: DISTINCT on large column list ─────────────────────
def _distinct_many_columns(tree: exp.Select) -> list[dict]:
    if tree.args.get("distinct"):
        cols = list(tree.find_all(exp.Column))
        if len(cols) > 5:
            return [{
                "rule": "distinct_many_columns", "severity": "warning", "line": 0,
                "message": f"DISTINCT on {len(cols)} columns. Verify dedup is needed at this grain."
            }]
    return []


# ── Rule 8: ORDER BY without LIMIT ────────────────────────────
def _order_by_no_limit(tree: exp.Select) -> list[dict]:
    if tree.find(exp.Order) and not tree.find(exp.Limit):
        return [{
            "rule": "order_by_no_limit", "severity": "warning", "line": 0,
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
                    "rule": "group_by_ordinal", "severity": "warning", "line": 0,
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
                    "rule": "float_equality", "severity": "warning", "line": 0,
                    "message": "Floating-point equality comparison. Use ABS(a - b) < threshold."
                })
                break
    return findings


# ── Rule 11: HAVING without GROUP BY ──────────────────────────
def _having_without_group_by(tree: exp.Select) -> list[dict]:
    if tree.find(exp.Having) and not tree.args.get("group"):
        return [{
            "rule": "having_without_group_by", "severity": "error", "line": 0,
            "message": "HAVING without GROUP BY. HAVING filters groups — use WHERE if no aggregation."
        }]
    return []


# ── Rule 12: Subquery without alias ───────────────────────────
def _subquery_no_alias(tree: exp.Select) -> list[dict]:
    findings = []
    for subq in tree.find_all(exp.Subquery):
        if not subq.args.get("alias"):
            findings.append({
                "rule": "subquery_no_alias", "severity": "error", "line": 0,
                "message": "Subquery missing alias. Every subquery in FROM needs an alias."
            })
    return findings


# ── Rule 13: UNION without ALL (dedup cost) ───────────────────
def _union_no_all(tree: exp.Select) -> list[dict]:
    findings = []
    for union in tree.find_all(exp.Union):
        if union.kind in ("DISTINCT", None, ""):
            findings.append({
                "rule": "union_no_all", "severity": "info", "line": 0,
                "message": "UNION (implicit DISTINCT) sorts dedup. Use UNION ALL if duplicates are acceptable."
            })
    return findings


# ── Rule 14: COUNT(DISTINCT ...) on many rows ─────────────────
def _count_distinct_many(tree: exp.Select) -> list[dict]:
    findings = []
    for agg in tree.find_all(exp.Count):
        if agg.args.get("distinct"):
            findings.append({
                "rule": "count_distinct_many", "severity": "info", "line": 0,
                "message": "COUNT(DISTINCT ...) is expensive on large datasets. Consider approximate methods (APPROX_COUNT_DISTINCT)."
            })
    return findings


# ── Rule 15: OR in JOIN condition ─────────────────────────────
def _or_in_join(tree: exp.Select) -> list[dict]:
    findings = []
    for join in tree.find_all(exp.Join):
        for or_ in join.find_all(exp.Or):
            findings.append({
                "rule": "or_in_join", "severity": "warning", "line": 0,
                "message": "OR in JOIN condition prevents hash join. Consider UNION of two queries."
            })
    return findings


# ── Rule 16: IN with many values ──────────────────────────────
def _in_many_values(tree: exp.Select) -> list[dict]:
    findings = []
    for in_ in tree.find_all(exp.In):
        if isinstance(in_.this, exp.Column) and in_.find(exp.Subquery) is None:
            values = [v for v in in_.expressions if isinstance(v, exp.Literal)]
            if len(values) > 10:
                findings.append({
                    "rule": "in_many_values", "severity": "info", "line": 0,
                    "message": f"IN list with {len(values)} values. Consider a temp table or JOIN."
                })
    return findings


# ── Rule 17: Nested aggregate (aggregate of aggregate) ────────
def _nested_aggregate(tree: exp.Select) -> list[dict]:
    findings = []
    for agg in tree.find_all(exp.AggFunc):
        # Check if any child AggFunc is NOT itself (i.e., a nested aggregate)
        has_nested = any(child is not agg and isinstance(child, exp.AggFunc)
                        for child in agg.find_all(exp.AggFunc))
        if has_nested:
            findings.append({
                "rule": "nested_aggregate", "severity": "warning", "line": 0,
                "message": f"Nested aggregate ({agg.sql()}). Use a window function or CTE instead."
            })
    return findings


# ── Rule 18: No filter on ID column ───────────────────────────
def _no_filter_on_id(tree: exp.Select) -> list[dict]:
    """Flag SELECT from a single table with no WHERE filtering an *_id column."""
    if not tree.find(exp.Where):
        return []
    from_tables = list(tree.find_all(exp.Table))
    if len(from_tables) != 1:
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
            "rule": "no_filter_on_id", "severity": "info", "line": 0,
            "message": "WHERE clause present but no filter on *_id column. Verify scan is intended."
        }]
    return []


# ── Rule 19: Correlated subquery ──────────────────────────────
def _correlated_subquery(tree: exp.Select) -> list[dict]:
    findings = []
    for subq in tree.find_all(exp.Subquery):
        inner_tree = subq.this
        outer_names: set[str] = set()
        for t in tree.find_all(exp.Table):
            if t.find_ancestor(exp.Subquery) is None:
                outer_names.add(t.name)
                alias = t.args.get("alias")
                if alias and alias.this:
                    outer_names.add(alias.this.name)
        for col in inner_tree.find_all(exp.Column):
            if col.table and col.table in outer_names:
                findings.append({
                    "rule": "correlated_subquery", "severity": "warning", "line": 0,
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