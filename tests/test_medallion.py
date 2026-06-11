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
        assert len(result["silver"]["issues"]) > 0
        conn.close()