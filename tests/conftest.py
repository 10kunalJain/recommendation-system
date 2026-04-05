"""Shared test fixtures for the recommendation system."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def sample_articles():
    """Minimal article dataset for testing."""
    return pd.DataFrame({
        "article_id": ["001", "002", "003", "004", "005", "006", "007", "008"],
        "product_type_name": ["T-shirt", "Trousers", "Dress", "Jacket", "Socks", "T-shirt", "Dress", "Jacket"],
        "colour_group_name": ["Black", "Blue", "Red", "Black", "White", "White", "Black", "Blue"],
        "section_name": ["Menswear", "Menswear", "Womenswear", "Menswear", "Menswear", "Womenswear", "Womenswear", "Womenswear"],
    })


@pytest.fixture(scope="session")
def sample_transactions():
    """Minimal transaction dataset with temporal structure."""
    np.random.seed(42)
    dates = pd.date_range("2020-07-01", "2020-09-15", freq="D")

    rows = []
    customers = [f"user_{i}" for i in range(10)]
    articles = ["001", "002", "003", "004", "005", "006", "007", "008"]

    for _ in range(200):
        rows.append({
            "t_dat": np.random.choice(dates),
            "customer_id": np.random.choice(customers),
            "article_id": np.random.choice(articles),
        })

    df = pd.DataFrame(rows)
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    df["article_id"] = df["article_id"].astype(str)
    return df


@pytest.fixture(scope="session")
def sample_customers():
    """Minimal customer dataset."""
    return pd.DataFrame({
        "customer_id": [f"user_{i}" for i in range(10)],
        "age": [22, 35, 45, 28, 55, 19, 33, 41, 60, 30],
    })


@pytest.fixture(scope="session")
def train_val_test_split(sample_transactions):
    """Temporal split of sample transactions."""
    train_end = pd.Timestamp("2020-09-01")
    val_end = pd.Timestamp("2020-09-08")

    train = sample_transactions[sample_transactions["t_dat"] < train_end].copy()
    val = sample_transactions[
        (sample_transactions["t_dat"] >= train_end)
        & (sample_transactions["t_dat"] < val_end)
    ].copy()
    test = sample_transactions[sample_transactions["t_dat"] >= val_end].copy()
    return train, val, test


@pytest.fixture(scope="session")
def user_item_mappings(train_val_test_split):
    """User/item index mappings from training data."""
    train = train_val_test_split[0]
    users = train["customer_id"].unique()
    items = train["article_id"].unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    item_to_idx = {a: i for i, a in enumerate(items)}
    idx_to_item = {i: a for a, i in item_to_idx.items()}
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item
