"""Deterministic sample retail dataset with seeded dirt + anomalies."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
WEEKS = 78  # enough history for 8-week rolling baselines
SPIKE_WEEK = WEEKS - 3   # GDV x4 spike
DROP_WEEK = WEEKS - 10   # GDV -60% drop


def generate(out_dir: Path = DATA_DIR) -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = pd.DataFrame({
        "customer_id": np.arange(1, 501),
        "name": [f"Customer {i}" for i in range(1, 501)],
        "email": [f"c{i}@example.com" for i in range(1, 501)],
        "country": rng.choice(["US", "IN", "DE", "UK"], 500, p=[.5, .2, .15, .15]),
        "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, 500), "D"),
    })
    customers.loc[rng.choice(500, 15, replace=False), "email"] = None  # dirty: null emails
    customers = pd.concat([customers, customers.iloc[:10]], ignore_index=True)  # dirty: dup rows

    products = pd.DataFrame({
        "product_id": np.arange(1, 61),
        "product_name": [f"Product {i}" for i in range(1, 61)],
        "category": rng.choice(["Electronics", "Home", "Toys", "Office"], 60),
        "list_price": np.round(rng.uniform(5, 400, 60), 2),
    })

    start = pd.Timestamp("2024-06-03")  # a Monday
    rows = []
    oid = 1
    for w in range(WEEKS):
        base = 60 + 10 * np.sin(w / 6)
        mult = 4.0 if w == SPIKE_WEEK else (0.4 if w == DROP_WEEK else 1.0)
        n = int(rng.poisson(base * mult))
        for _ in range(n):
            day = start + pd.Timedelta(weeks=w, days=int(rng.integers(0, 7)))
            pid = int(rng.integers(1, 61))
            price = float(products.loc[pid - 1, "list_price"]) * float(rng.uniform(.9, 1.1))
            rows.append((oid, int(rng.integers(1, 501)), pid, day.date().isoformat(),
                         int(rng.integers(1, 5)),
                         round(price * mult if w == SPIKE_WEEK else price, 2),
                         str(rng.choice(["web", "mobile", "partner"]))))
            oid += 1
    orders = pd.DataFrame(rows, columns=["order_id", "customer_id", "product_id",
                                         "order_date", "quantity", "unit_price", "channel"])
    dirty_idx = rng.choice(len(orders), 25, replace=False)
    orders.loc[dirty_idx[:20], "quantity"] = -1                      # dirty: negative qty
    orders["unit_price"] = orders["unit_price"].astype(object)
    orders.loc[dirty_idx[20:], "unit_price"] = "N/A"                 # dirty: bad price strings

    customers.to_csv(out_dir / "customers.csv", index=False)
    products.to_csv(out_dir / "products.csv", index=False)
    orders.to_csv(out_dir / "orders.csv", index=False)


if __name__ == "__main__":
    generate()
    print(f"Sample data written to {DATA_DIR}")
