"""Gold layer — aggregated, analytics-ready marts."""
from __future__ import annotations
import duckdb


def build_gold_marts(conn: duckdb.DuckDBPyConnection) -> dict:
    """Build gold-layer aggregate marts from silver tables.
    Creates: weekly_sales, customer_metrics, category_performance.
    Returns descriptions + row counts.
    """
    marts = {}

    # Weekly Sales Mart
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

    # Customer Metrics Mart
    conn.execute("""
        CREATE OR REPLACE TABLE gold_customer_metrics AS
        SELECT
            c.customer_id, c.name, c.country,
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

    # Category Performance Mart
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