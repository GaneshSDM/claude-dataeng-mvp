"""Silver layer — cleaning, dedup, validation, and type casting."""
from __future__ import annotations
import duckdb


def run_silver(conn: duckdb.DuckDBPyConnection) -> dict:
    """Clean bronze tables into silver layer.
    - orders: remove negative quantities, parse prices, normalize dates
    - customers: deduplicate, fill null emails
    - products: validate category values
    Returns quality report: {cleaning_steps, counts, issues}
    """
    steps = []
    issues = []

    # Clean orders
    conn.execute("""
        CREATE OR REPLACE TABLE silver_orders AS
        SELECT
            order_id, customer_id, product_id,
            CAST(order_date AS DATE) AS order_date,
            CASE WHEN quantity > 0 THEN quantity ELSE NULL END AS quantity,
            CASE WHEN TRY_CAST(unit_price AS DOUBLE) IS NOT NULL
                 THEN CAST(unit_price AS DOUBLE) ELSE NULL END AS unit_price,
            channel
        FROM bronze_orders
    """)
    total_orders = conn.execute("SELECT COUNT(*) FROM bronze_orders").fetchone()[0]
    bad_qty = conn.execute("SELECT COUNT(*) FROM silver_orders WHERE quantity IS NULL").fetchone()[0]
    bad_price = conn.execute("SELECT COUNT(*) FROM silver_orders WHERE unit_price IS NULL").fetchone()[0]
    null_dates = conn.execute("SELECT COUNT(*) FROM silver_orders WHERE order_date IS NULL").fetchone()[0]
    steps.append(f"Total raw orders: {total_orders}")
    steps.append(f"Negative/null quantities cleaned: {bad_qty}")
    steps.append(f"Non-numeric prices cleaned: {bad_price}")
    steps.append(f"Null dates found: {null_dates}")
    if bad_qty > 0 or bad_price > 0:
        issues.append(f"Data quality: {bad_qty} bad quantities, {bad_price} bad prices")

    # Clean customers
    conn.execute("""
        CREATE OR REPLACE TABLE silver_customers AS
        SELECT DISTINCT customer_id, name,
            COALESCE(email, 'unknown@unknown.com') AS email,
            country, signup_date
        FROM bronze_customers
    """)
    orig_cust = conn.execute("SELECT COUNT(*) FROM bronze_customers").fetchone()[0]
    deduped = conn.execute("SELECT COUNT(*) FROM silver_customers").fetchone()[0]
    dupes = orig_cust - deduped
    if dupes > 0:
        steps.append(f"Customers deduplicated: {dupes} duplicates removed")
        issues.append(f"Data quality: {dupes} duplicate customer rows")

    # Clean products
    conn.execute("""
        CREATE OR REPLACE TABLE silver_products AS
        SELECT product_id, product_name,
            CASE WHEN category IN ('Electronics', 'Home', 'Toys', 'Office')
                 THEN category ELSE 'Other' END AS category,
            list_price
        FROM bronze_products
    """)
    bad_cats = conn.execute("""SELECT COUNT(*) FROM bronze_products
        WHERE category NOT IN ('Electronics', 'Home', 'Toys', 'Office')""").fetchone()[0]
    if bad_cats > 0:
        steps.append(f"Products with invalid categories reclassified: {bad_cats}")
        issues.append(f"Data quality: {bad_cats} products had invalid categories")

    return {
        "steps": steps,
        "issues": issues,
        "silver_tables": ["silver_orders", "silver_customers", "silver_products"],
        "row_counts": {
            "silver_orders": conn.execute("SELECT COUNT(*) FROM silver_orders").fetchone()[0],
            "silver_customers": conn.execute("SELECT COUNT(*) FROM silver_customers").fetchone()[0],
            "silver_products": conn.execute("SELECT COUNT(*) FROM silver_products").fetchone()[0],
        },
    }