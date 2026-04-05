"""Data loading and temporal splitting for H&M dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from src.utils.config import CONFIG


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files into DataFrames."""
    raw_dir = Path(CONFIG["data"]["raw_dir"])

    logger.info("Loading raw data...")
    articles = pd.read_csv(raw_dir / CONFIG["data"]["articles_file"])
    transactions = pd.read_csv(raw_dir / CONFIG["data"]["transactions_file"])
    customers = pd.read_csv(raw_dir / CONFIG["data"]["customers_file"])

    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["article_id"] = transactions["article_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)

    logger.info(
        f"Loaded {len(articles)} articles, {len(transactions)} transactions, "
        f"{len(customers)} customers"
    )
    return articles, transactions, customers


def temporal_split(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split transactions by time into train/val/test.

    This mirrors real-world deployment: we train on past, predict future.
    """
    train_end = pd.Timestamp(CONFIG["data"]["train_end_date"])
    val_end = pd.Timestamp(CONFIG["data"]["val_end_date"])

    train = transactions[transactions["t_dat"] < train_end].copy()
    val = transactions[
        (transactions["t_dat"] >= train_end) & (transactions["t_dat"] < val_end)
    ].copy()
    test = transactions[transactions["t_dat"] >= val_end].copy()

    logger.info(
        f"Temporal split: train={len(train)}, val={len(val)}, test={len(test)}"
    )
    return train, val, test


def build_user_item_mappings(
    transactions: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    """Create bidirectional mappings between user/item IDs and integer indices.

    Required for sparse matrix operations in ALS and embedding models.
    """
    unique_users = transactions["customer_id"].unique()
    unique_items = transactions["article_id"].unique()

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}
    item_to_idx = {iid: i for i, iid in enumerate(unique_items)}
    idx_to_item = {i: iid for iid, i in item_to_idx.items()}

    logger.info(f"Mappings: {len(user_to_idx)} users, {len(item_to_idx)} items")
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def build_interaction_matrix(
    transactions: pd.DataFrame,
    user_to_idx: dict,
    item_to_idx: dict,
) -> "scipy.sparse.csr_matrix":
    """Build sparse user-item interaction matrix with confidence weighting.

    Uses log(1 + count) as implicit confidence signal — accounts for
    repeat purchases without letting power buyers dominate.
    """
    from scipy.sparse import csr_matrix
    from collections import Counter

    interactions = Counter(
        zip(transactions["customer_id"], transactions["article_id"])
    )

    rows, cols, data = [], [], []
    for (user, item), count in interactions.items():
        if user in user_to_idx and item in item_to_idx:
            rows.append(user_to_idx[user])
            cols.append(item_to_idx[item])
            data.append(np.log1p(count))  # confidence weighting

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    logger.info(
        f"Interaction matrix: {matrix.shape}, nnz={matrix.nnz}, "
        f"density={matrix.nnz / (n_users * n_items):.6f}"
    )
    return matrix
