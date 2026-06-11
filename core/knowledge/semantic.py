"""Semantic layer — YAML-defined business metrics that resolve to SQL."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricDef:
    name: str
    description: str
    sql: str
    type: str          # "simple" | "aggregate" | "ratio" | "derived"
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    source_table: str = ""
    dimensions: Optional[list[str]] = None


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
                sql="""SELECT ROUND(100.0 * SUM(CASE WHEN cnt > 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS repeat_customer_rate FROM (SELECT customer_id, COUNT(*) AS cnt FROM silver_orders GROUP BY customer_id)""",
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
        return [{"name": m.name, "description": m.description, "type": m.type}
                for m in self._metrics.values()]

    def get_metric(self, name: str) -> Optional[MetricDef]:
        return self._metrics.get(name)

    def resolve(self, metric_name: str, dimensions: Optional[list[str]] = None) -> str:
        """Resolve a metric name + optional dimensions to executable SQL."""
        metric = self._metrics.get(metric_name)
        if metric is None:
            return f"-- Unknown metric: {metric_name}"
        return metric.sql

    def from_yaml(self, yaml_str: str) -> int:
        """Load metric definitions from a YAML string. Returns number loaded."""
        try:
            import yaml
            data = yaml.safe_load(yaml_str)
            if not data or "metrics" not in data:
                return 0
            count = 0
            for m in data["metrics"]:
                self._metrics[m["name"]] = MetricDef(
                    name=m["name"], description=m.get("description", ""),
                    sql=m["sql"], type=m.get("type", "simple"),
                    source_table=m.get("source_table", ""),
                    dimensions=m.get("dimensions", []),
                )
                count += 1
            return count
        except Exception:
            return 0