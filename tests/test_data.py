import pandas as pd
from data.generate_sample import generate


def test_generate_creates_seeded_retail_data(tmp_path):
    generate(tmp_path)
    orders = pd.read_csv(tmp_path / "orders.csv")
    customers = pd.read_csv(tmp_path / "customers.csv")
    products = pd.read_csv(tmp_path / "products.csv")
    assert len(orders) > 4000 and len(customers) == 510 and len(products) == 60
    # dirty rows seeded
    assert (orders["quantity"] < 0).sum() >= 10
    # "N/A" price strings in CSV surface as NaN under pandas default na_values
    assert orders["unit_price"].isna().sum() >= 5
    assert customers["email"].isna().sum() >= 10
    # deterministic
    generate(tmp_path)
    orders2 = pd.read_csv(tmp_path / "orders.csv")
    assert orders.equals(orders2)
