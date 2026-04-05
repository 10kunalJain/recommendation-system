"""Feature engineering for the ranking model.

This module creates user-level, item-level, and user-item interaction features
that capture behavioral signals beyond raw collaborative filtering scores.
"""

import pandas as pd
import numpy as np
from loguru import logger


def build_user_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute user-level behavioral features.

    These features capture purchasing patterns that help the ranking model
    understand user preferences and activity levels.
    """
    logger.info("Building user features...")
    txn = transactions.merge(articles, on="article_id", how="left")

    uf = txn.groupby("customer_id").agg(
        purchase_count=("article_id", "count"),
        unique_items=("article_id", "nunique"),
        last_purchase_date=("t_dat", "max"),
        first_purchase_date=("t_dat", "min"),
        unique_sections=("section_name", "nunique"),
        unique_colors=("colour_group_name", "nunique"),
    ).reset_index()

    # Recency: days since last purchase (critical for fashion — trends fade fast)
    uf["purchase_recency_days"] = (
        reference_date - uf["last_purchase_date"]
    ).dt.days

    # Frequency: avg purchases per active day
    active_days = (uf["last_purchase_date"] - uf["first_purchase_date"]).dt.days + 1
    uf["purchase_frequency"] = uf["purchase_count"] / active_days.clip(lower=1)

    # Color diversity: how eclectic is the user?
    uf["color_diversity"] = uf["unique_colors"] / uf["purchase_count"].clip(lower=1)

    # Favorite section per user
    fav_section = (
        txn.groupby(["customer_id", "section_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates("customer_id")
        .rename(columns={"section_name": "favorite_section"})
        [["customer_id", "favorite_section"]]
    )
    uf = uf.merge(fav_section, on="customer_id", how="left")

    # Merge age from customers
    uf = uf.merge(customers, on="customer_id", how="left")
    uf["age"] = uf["age"].fillna(uf["age"].median())
    uf["age_group"] = pd.cut(
        uf["age"], bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"]
    )

    drop_cols = ["last_purchase_date", "first_purchase_date"]
    uf.drop(columns=drop_cols, inplace=True)

    logger.info(f"User features: {uf.shape}")
    return uf


def build_item_features(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute item-level features.

    Captures item popularity, buyer demographics, and lifecycle stage.
    """
    logger.info("Building item features...")

    txn = transactions.merge(
        articles[["article_id"]], on="article_id", how="inner"
    )

    itf = txn.groupby("article_id").agg(
        total_purchases=("customer_id", "count"),
        unique_buyers=("customer_id", "nunique"),
        first_purchase=("t_dat", "min"),
        last_purchase=("t_dat", "max"),
    ).reset_index()

    # Repurchase rate: signal of item stickiness
    itf["repurchase_rate"] = (
        (itf["total_purchases"] - itf["unique_buyers"])
        / itf["total_purchases"].clip(lower=1)
    )

    itf["days_since_first_purchase"] = (
        reference_date - itf["first_purchase"]
    ).dt.days
    itf["days_since_last_purchase"] = (
        reference_date - itf["last_purchase"]
    ).dt.days

    # Section popularity rank
    itf = itf.merge(articles, on="article_id", how="left")
    section_rank = (
        itf.groupby("section_name")["total_purchases"]
        .rank(ascending=False, method="dense")
    )
    itf["section_popularity_rank"] = section_rank

    # Average buyer age (proxy for item demographic targeting)
    customer_ages = txn.merge(
        transactions[["customer_id"]].drop_duplicates(),
        on="customer_id",
    )
    buyer_age = (
        txn.groupby("article_id")["customer_id"]
        .apply(lambda x: len(x))  # placeholder — real impl merges age
        .reset_index(name="buyer_count_check")
    )

    itf.drop(columns=["first_purchase", "last_purchase"], inplace=True)
    logger.info(f"Item features: {itf.shape}")
    return itf


def build_interaction_features(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    user_features: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build pairwise user-item affinity features.

    These features are computed for each (user, candidate_item) pair
    and capture how well an item matches a user's historical preferences.
    """
    logger.info("Building interaction features...")
    txn = transactions.merge(articles, on="article_id", how="left")

    # User-section affinity: fraction of purchases in each section
    user_section = (
        txn.groupby(["customer_id", "section_name"])
        .size()
        .reset_index(name="section_count")
    )
    user_total = txn.groupby("customer_id").size().reset_index(name="total_count")
    user_section = user_section.merge(user_total, on="customer_id")
    user_section["user_section_affinity"] = (
        user_section["section_count"] / user_section["total_count"]
    )

    # User-color affinity
    user_color = (
        txn.groupby(["customer_id", "colour_group_name"])
        .size()
        .reset_index(name="color_count")
    )
    user_color = user_color.merge(user_total, on="customer_id")
    user_color["user_color_affinity"] = (
        user_color["color_count"] / user_color["total_count"]
    )

    # User-product_type affinity
    user_ptype = (
        txn.groupby(["customer_id", "product_type_name"])
        .size()
        .reset_index(name="ptype_count")
    )
    user_ptype = user_ptype.merge(user_total, on="customer_id")
    user_ptype["user_product_type_affinity"] = (
        user_ptype["ptype_count"] / user_ptype["total_count"]
    )

    # Co-visitation matrix: items frequently bought together
    # (within same user session = same day)
    covisit = _build_covisitation_scores(txn)

    logger.info("Interaction features built.")
    return {
        "user_section_affinity": user_section[
            ["customer_id", "section_name", "user_section_affinity"]
        ],
        "user_color_affinity": user_color[
            ["customer_id", "colour_group_name", "user_color_affinity"]
        ],
        "user_product_type_affinity": user_ptype[
            ["customer_id", "product_type_name", "user_product_type_affinity"]
        ],
        "covisitation": covisit,
    }


def _build_covisitation_scores(txn: pd.DataFrame) -> dict:
    """Build item-item co-visitation scores.

    Two items are "co-visited" if the same user bought them on the same day.
    This captures complementary item relationships (e.g., top + bottom).
    """
    logger.info("Computing co-visitation scores...")
    txn["session_key"] = (
        txn["customer_id"].astype(str) + "_" + txn["t_dat"].dt.date.astype(str)
    )

    # Group items by session
    session_items = txn.groupby("session_key")["article_id"].apply(list)

    from collections import Counter
    covisit_counts = Counter()
    for items in session_items:
        unique_items = list(set(items))
        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                pair = tuple(sorted([unique_items[i], unique_items[j]]))
                covisit_counts[pair] += 1

    # Normalize to scores
    if covisit_counts:
        max_count = max(covisit_counts.values())
        covisit_scores = {
            pair: count / max_count for pair, count in covisit_counts.items()
        }
    else:
        covisit_scores = {}

    logger.info(f"Co-visitation pairs: {len(covisit_scores)}")
    return covisit_scores


def assemble_ranking_features(
    candidates_df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_features: dict,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble final feature matrix for ranking model.

    Joins user features, item features, and interaction features for
    each (user, candidate_item) pair.
    """
    logger.info("Assembling ranking features...")
    df = candidates_df.copy()

    # Merge user features
    uf_cols = [
        "customer_id", "purchase_count", "unique_items",
        "purchase_recency_days", "purchase_frequency", "color_diversity", "age",
    ]
    available_uf_cols = [c for c in uf_cols if c in user_features.columns]
    df = df.merge(user_features[available_uf_cols], on="customer_id", how="left")

    # Merge item features
    itf_cols = [
        "article_id", "total_purchases", "unique_buyers",
        "repurchase_rate", "days_since_first_purchase", "section_popularity_rank",
    ]
    available_itf_cols = [c for c in itf_cols if c in item_features.columns]
    df = df.merge(item_features[available_itf_cols], on="article_id", how="left")

    # Merge article metadata for interaction feature lookups
    df = df.merge(articles, on="article_id", how="left")

    # Lookup user-section affinity
    section_aff = interaction_features["user_section_affinity"]
    df = df.merge(
        section_aff,
        on=["customer_id", "section_name"],
        how="left",
    )

    # Lookup user-color affinity
    color_aff = interaction_features["user_color_affinity"]
    df = df.merge(
        color_aff,
        on=["customer_id", "colour_group_name"],
        how="left",
    )

    # Lookup user-product_type affinity
    ptype_aff = interaction_features["user_product_type_affinity"]
    df = df.merge(
        ptype_aff,
        on=["customer_id", "product_type_name"],
        how="left",
    )

    # Fill missing affinities with 0 (new combinations)
    for col in ["user_section_affinity", "user_color_affinity", "user_product_type_affinity"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Fill other NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    logger.info(f"Ranking feature matrix: {df.shape}")
    return df
