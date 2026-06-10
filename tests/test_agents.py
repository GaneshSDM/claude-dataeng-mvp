"""Tests for the multi-agent system."""

import pytest
from pathlib import Path
from data.generate_sample import generate
from backend.agents.sql_agent import SQLAgent
from backend.agents.etl_agent import ETLAgent
from backend.agents.quality_agent import QualityAgent
from backend.agents.analytics_agent import AnalyticsAgent
from backend.agents.doc_agent import DocumentationAgent
from backend.agents.orchestrator import OrchestratorAgent


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(d)
    return d


class TestSQLAgent:
    def test_top_products_by_revenue(self, data_dir):
        agent = SQLAgent(data_dir)
        result = agent.run("Show me top 5 products by revenue")
        assert result.status == "ok"
        assert result.sql is not None
        assert "SELECT" in result.sql.upper()
        assert result.artifacts is not None

    def test_monthly_revenue_trend(self, data_dir):
        agent = SQLAgent(data_dir)
        result = agent.run("What's the monthly revenue trend?")
        assert result.status == "ok"
        assert result.sql is not None
        assert "month" in result.sql.lower() or "strftime" in result.sql.lower()

    def test_repeat_customer_rate(self, data_dir):
        agent = SQLAgent(data_dir)
        result = agent.run("What's our repeat customer rate?")
        assert result.status == "ok"
        assert "repeat" in result.detail.lower()

    def test_total_customers(self, data_dir):
        agent = SQLAgent(data_dir)
        result = agent.run("How many customers do we have?")
        assert result.status == "ok"
        assert result.sql is not None

    def test_revenue_by_channel(self, data_dir):
        agent = SQLAgent(data_dir)
        result = agent.run("Show me revenue by channel")
        assert result.status == "ok"
        assert "channel" in result.sql.lower()

    def test_unknown_query_falls_through(self, data_dir):
        agent = SQLAgent(data_dir)
        result = agent.run("What is the meaning of life?")
        # Should handle gracefully
        assert result.status in ("ok", "error")


class TestETLAgent:
    def test_full_etl_lifecycle(self, data_dir):
        agent = ETLAgent(data_dir)
        result = agent.run("Run full ETL lifecycle")
        assert result.status == "ok"

    def test_profile_only(self, data_dir):
        agent = ETLAgent(data_dir)
        result = agent.run("Profile the data please")
        assert result.status == "ok"
        assert "orders" in result.detail.lower()

    def test_clean_only(self, data_dir):
        agent = ETLAgent(data_dir)
        result = agent.run("Clean the data")
        assert result.status == "ok"

    def test_default_lifecycle(self, data_dir):
        agent = ETLAgent(data_dir)
        result = agent.run("go")
        assert result.status == "ok"


class TestQualityAgent:
    def test_anomaly_scan_finds_seeded(self, data_dir):
        agent = QualityAgent(data_dir)
        result = agent.run("Scan for anomalies")
        assert result.status == "ok"
        findings = result.artifacts[0]["findings"] if result.artifacts else []
        # Should find at least the seeded spike or drop
        assert any(f["type"] == "anomaly" for f in findings)

    def test_full_quality_scan(self, data_dir):
        agent = QualityAgent(data_dir)
        result = agent.run("Run full quality scan including PII")
        assert result.status == "ok"
        assert len(result.detail) > 100

    def test_pii_scan(self, data_dir):
        agent = QualityAgent(data_dir)
        result = agent.run("Scan for PII")
        assert result.status == "ok"

    def test_quality_profile(self, data_dir):
        agent = QualityAgent(data_dir)
        result = agent.run("Profile data quality")
        assert result.status == "ok"


class TestAnalyticsAgent:
    def test_executive_summary(self, data_dir):
        agent = AnalyticsAgent(data_dir)
        result = agent.run("Executive summary")
        assert result.status == "ok"
        assert "Revenue" in result.detail or "revenue" in result.detail.lower()

    def test_kpi_report(self, data_dir):
        agent = AnalyticsAgent(data_dir)
        result = agent.run("Show me KPIs and metrics")
        assert result.status == "ok"
        assert "KPI" in result.detail or "Revenue" in result.detail

    def test_segmentation(self, data_dir):
        agent = AnalyticsAgent(data_dir)
        result = agent.run("Compare revenue by country and channel")
        assert result.status == "ok"
        assert any(w in result.detail for w in ["Country", "country", "Channel", "channel", "Revenue"])

    def test_report_generates_charts(self, data_dir):
        agent = AnalyticsAgent(data_dir)
        result = agent.run("Executive summary with charts")
        assert result.status == "ok"
        assert any(a["type"] == "chart" for a in result.artifacts)


class TestDocumentationAgent:
    def test_data_dictionary(self, data_dir):
        agent = DocumentationAgent(data_dir)
        result = agent.run("Generate data dictionary")
        assert result.status == "ok"
        assert "orders" in result.detail.lower()
        assert "customers" in result.detail.lower()

    def test_schema_documentation(self, data_dir):
        agent = DocumentationAgent(data_dir)
        result = agent.run("Describe the schema")
        assert result.status == "ok"
        assert "Schema" in result.detail or "schema" in result.detail.lower()

    def test_pipeline_documentation(self, data_dir):
        agent = DocumentationAgent(data_dir)
        result = agent.run("Generate pipeline and lineage documentation")
        assert result.status == "ok"
        assert "Pipeline" in result.detail or "pipeline" in result.detail.lower()


class TestOrchestrator:
    def test_list_capabilities(self, data_dir):
        orch = OrchestratorAgent(data_dir)
        agents = orch.list_capabilities()
        names = [a["name"] for a in agents]
        assert "sql" in names
        assert "etl" in names
        assert "quality" in names
        assert "analytics" in names
        assert "docs" in names

    def test_routes_to_sql_agent(self, data_dir):
        orch = OrchestratorAgent(data_dir)
        result = orch.run("Show me top products by revenue")
        assert result.status == "ok"
        assert result.agent == "sql" or "sql" in result.detail.lower()

    def test_routes_to_quality(self, data_dir):
        orch = OrchestratorAgent(data_dir)
        result = orch.run("Scan for anomalies and PII")
        assert result.status == "ok"

    def test_routes_to_analytics(self, data_dir):
        orch = OrchestratorAgent(data_dir)
        result = orch.run("Executive summary with KPIs")
        assert result.status == "ok"

    def test_multi_agent_routing(self, data_dir):
        orch = OrchestratorAgent(data_dir)
        result = orch.run("Run everything - all agents full lifecycle")
        assert result.status in ("ok", "partial")
        # Should have run multiple agents
        assert "agents" in result.detail.lower() or len(result.detail) > 500

    def test_get_agent_by_name(self, data_dir):
        orch = OrchestratorAgent(data_dir)
        sql = orch.get_agent("sql")
        assert sql is not None
        assert sql.name == "sql"
        assert orch.get_agent("nonexistent") is None