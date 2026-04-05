"""Baseline Evaluators.

Runs each retrieval model independently to establish baselines.
This is CRITICAL for the impact narrative — without baselines,
a MAP@12 of 0.005 means nothing. With baselines, "2.5x over
popularity" tells a clear story.
"""

import numpy as np
import pandas as pd
from loguru import logger
from src.evaluation.metrics import evaluate_recommendations


def evaluate_popularity_baseline(
    popularity_generator,
    ground_truth: dict[str, set[str]],
    customers: pd.DataFrame,
    k: int = 12,
) -> dict[str, float]:
    """Evaluate popularity-only baseline (no personalization)."""
    logger.info("Evaluating: Popularity Baseline")
    predictions = {}
    for customer_id in ground_truth.keys():
        # Try age-group specific popularity
        cust_row = customers[customers["customer_id"] == customer_id]
        age_group = None
        if len(cust_row) > 0:
            age = cust_row.iloc[0].get("age")
            if pd.notna(age):
                if age <= 25: age_group = "18-25"
                elif age <= 35: age_group = "26-35"
                elif age <= 45: age_group = "36-45"
                elif age <= 55: age_group = "46-55"
                else: age_group = "55+"

        recs = popularity_generator.generate_candidates(n_candidates=k, age_group=age_group)
        predictions[customer_id] = [aid for aid, _ in recs]

    return evaluate_recommendations(predictions, ground_truth, [5, 12, 20])


def evaluate_als_baseline(
    als_generator,
    user_to_idx: dict,
    idx_to_item: dict,
    ground_truth: dict[str, set[str]],
    k: int = 12,
) -> dict[str, float]:
    """Evaluate ALS-only baseline (collaborative filtering without ranking)."""
    logger.info("Evaluating: ALS Baseline")
    predictions = {}
    for customer_id in ground_truth.keys():
        if customer_id not in user_to_idx:
            predictions[customer_id] = []
            continue
        user_idx = user_to_idx[customer_id]
        raw = als_generator.generate_candidates(user_idx, n_candidates=k)
        predictions[customer_id] = [
            idx_to_item[idx] for idx, _ in raw if idx in idx_to_item
        ]

    return evaluate_recommendations(predictions, ground_truth, [5, 12, 20])


def evaluate_content_baseline(
    content_generator,
    user_histories: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int = 12,
) -> dict[str, float]:
    """Evaluate content-based baseline."""
    logger.info("Evaluating: Content-Based Baseline")
    predictions = {}
    for customer_id in ground_truth.keys():
        history = user_histories.get(customer_id, [])
        if not history:
            predictions[customer_id] = []
            continue
        recs = content_generator.generate_candidates_for_user(history, n_candidates=k)
        predictions[customer_id] = [aid for aid, _ in recs]

    return evaluate_recommendations(predictions, ground_truth, [5, 12, 20])


def evaluate_recency_baseline(
    recency_generator,
    user_recent: dict[str, list[tuple[str, float]]],
    user_histories: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int = 12,
) -> dict[str, float]:
    """Evaluate recency-based baseline."""
    logger.info("Evaluating: Recency Baseline")
    predictions = {}
    for customer_id in ground_truth.keys():
        recent = user_recent.get(customer_id, [])
        history = set(user_histories.get(customer_id, []))
        if not recent:
            predictions[customer_id] = []
            continue
        recs = recency_generator.generate_candidates(recent, n_candidates=k, exclude=history)
        predictions[customer_id] = [aid for aid, _ in recs]

    return evaluate_recommendations(predictions, ground_truth, [5, 12, 20])


def run_all_baselines(pipeline, split: str = "test") -> dict[str, dict[str, float]]:
    """Run all baseline evaluations and return comparison table."""
    eval_txn = pipeline.test_txn if split == "test" else pipeline.val_txn
    ground_truth = (
        eval_txn.groupby("customer_id")["article_id"]
        .apply(set)
        .to_dict()
    )

    results = {}

    # 1. Popularity baseline
    results["Popularity"] = evaluate_popularity_baseline(
        pipeline.popularity_generator, ground_truth, pipeline.customers
    )

    # 2. ALS baseline
    results["ALS (CF)"] = evaluate_als_baseline(
        pipeline.als_generator, pipeline.user_to_idx,
        pipeline.idx_to_item, ground_truth
    )

    # 3. Content-based baseline
    results["Content-Based"] = evaluate_content_baseline(
        pipeline.content_generator, pipeline._user_history, ground_truth
    )

    # 4. Recency baseline
    results["Recency"] = evaluate_recency_baseline(
        pipeline.recency_generator, pipeline._user_recent,
        pipeline._user_history, ground_truth
    )

    # 5. Full hybrid pipeline
    logger.info("Evaluating: Full Hybrid Pipeline")
    results["Hybrid + Ranking"] = pipeline.evaluate(split=split)

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'MAP@12':<10} {'NDCG@12':<10} {'HR@12':<10} {'Recall@12':<10}")
    logger.info("-" * 70)
    for model_name, metrics in results.items():
        logger.info(
            f"{model_name:<20} "
            f"{metrics.get('map@12', 0):<10.4f} "
            f"{metrics.get('ndcg@12', 0):<10.4f} "
            f"{metrics.get('hit_rate@12', 0):<10.4f} "
            f"{metrics.get('recall@12', 0):<10.4f}"
        )
    logger.info("=" * 70)

    return results
