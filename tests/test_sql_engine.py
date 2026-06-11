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

    def test_correlated_subquery(self):
        findings = analyze_sql("SELECT o.order_id, (SELECT MAX(unit_price) FROM orders o2 WHERE o2.customer_id = o.customer_id) AS max_price FROM orders o")
        assert any(f["rule"] == "correlated_subquery" for f in findings)

    def test_negation_in_where(self):
        findings = analyze_sql("SELECT * FROM orders WHERE NOT quantity > 0")
        assert any(f["rule"] == "negation_in_where" for f in findings)

    def test_in_many_values(self):
        findings = analyze_sql("SELECT * FROM orders WHERE product_id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)")
        assert any(f["rule"] == "in_many_values" for f in findings)

    def test_union_no_all(self):
        findings = analyze_sql("SELECT product_id FROM products UNION SELECT product_id FROM orders")
        assert any(f["rule"] == "union_no_all" for f in findings)

    def test_nested_aggregate(self):
        findings = analyze_sql("SELECT MAX(COUNT(*)) FROM orders")
        assert any(f["rule"] == "nested_aggregate" for f in findings)

    def test_distinct_many_columns(self):
        findings = analyze_sql("SELECT DISTINCT order_id, customer_id, product_id, quantity, unit_price, channel FROM orders")
        assert any(f["rule"] == "distinct_many_columns" for f in findings)

    def test_no_filter_on_id(self):
        findings = analyze_sql("SELECT * FROM orders WHERE quantity > 0")
        assert any(f["rule"] == "no_filter_on_id" for f in findings)


class TestLineage:
    def test_simple_column(self):
        lineage = extract_lineage("SELECT order_id, quantity FROM orders")
        assert len(lineage) == 2
        assert lineage[0]["column"] == "order_id"
        assert lineage[0]["transformation_type"] == "direct"

    def test_aggregate_column(self):
        lineage = extract_lineage("SELECT COUNT(*) AS cnt FROM orders")
        assert any(l["column"] == "cnt" for l in lineage)

    def test_expression_column(self):
        lineage = extract_lineage("SELECT quantity * unit_price AS revenue FROM orders")
        assert any(l["column"] == "revenue" for l in lineage)