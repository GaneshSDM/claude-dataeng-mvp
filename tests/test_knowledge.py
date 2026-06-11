"""Tests for Knowledge Store and Semantic Registry."""
import pytest
from pathlib import Path
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
        assert isinstance(results, list)

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