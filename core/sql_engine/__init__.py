"""Deterministic SQL analysis engine — anti-pattern detection and column-level lineage."""

from core.sql_engine.rules import analyze_sql, SQLRule, RULES_REGISTRY
from core.sql_engine.lineage import extract_lineage

__all__ = ["analyze_sql", "extract_lineage", "SQLRule", "RULES_REGISTRY"]