import pytest
from data.generate_sample import generate
from backend.tools import ToolExecutor, TOOL_DEFS


@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(d)
    return ToolExecutor(d)


def test_tool_defs_shape():
    names = {t["name"] for t in TOOL_DEFS}
    assert names == {"profile_data", "run_sql", "run_anomaly_scan", "create_chart", "write_report"}
    assert all("input_schema" in t for t in TOOL_DEFS)


def test_profile_data(ex):
    out = ex.execute("profile_data", {"table": "orders"})
    assert out["result"]["row_count"] > 4000 and "quantity" in out["result"]["columns"]


def test_run_sql_select_and_ddl(ex):
    ex.execute("run_sql", {"query": "CREATE TABLE t1 AS SELECT 1 AS x", "purpose": "etl"})
    out = ex.execute("run_sql", {"query": "SELECT x FROM t1", "purpose": "question"})
    assert out["result"]["rows"] == [[1]]


def test_anomaly_scan_finds_seeded_anomalies(ex):
    out = ex.execute("run_anomaly_scan", {})
    dirs = {f["direction"] for f in out["result"]["findings"]}
    assert "SPIKE" in dirs or "DROP" in dirs


def test_chart_and_report_artifacts(ex):
    c = ex.execute("create_chart", {"title": "T", "chart_type": "bar",
                                    "x": ["a", "b"], "y": [1, 2],
                                    "x_label": "X", "y_label": "Y"})
    assert c["artifact"]["type"] == "chart" and c["artifact"]["x"] == ["a", "b"]
    r = ex.execute("write_report", {"markdown": "# Done"})
    assert r["artifact"]["type"] == "report"
