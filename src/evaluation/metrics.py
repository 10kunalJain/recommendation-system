"""Ranking Evaluation Metrics.

Production recommendation systems are evaluated on ranking quality,
not classification accuracy. A recommendation that's relevant but
ranked #50 is practically useless — position matters.

Implements: MAP@K, Recall@K, Precision@K, NDCG@K, Hit Rate@K, Coverage.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from loguru import logger


def precision_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """Fraction of top-K predictions that are relevant."""
    predicted_k = predicted[:k]
    if not predicted_k:
        return 0.0
    relevant = sum(1 for item in predicted_k if item in actual)
    return relevant / k


def recall_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """Fraction of relevant items captured in top-K."""
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    relevant = sum(1 for item in predicted_k if item in actual)
    return relevant / len(actual)


def average_precision_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """AP@K: average of precision values at each relevant position.

    This rewards placing relevant items higher in the ranking.
    """
    if not actual:
        return 0.0

    predicted_k = predicted[:k]
    score = 0.0
    num_hits = 0

    for i, item in enumerate(predicted_k):
        if item in actual:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(len(actual), k)


def ndcg_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Accounts for the position of relevant items using logarithmic discount.
    """
    predicted_k = predicted[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(predicted_k) if item in actual
    )

    # Ideal DCG: all relevant items at top positions
    n_relevant = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """Binary: did we get at least one relevant item in top-K?"""
    predicted_k = predicted[:k]
    return 1.0 if any(item in actual for item in predicted_k) else 0.0


def evaluate_recommendations(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate a full set of recommendations across all users.

    Args:
        predictions: {customer_id: [ranked list of article_ids]}
        ground_truth: {customer_id: {set of purchased article_ids}}
        k_values: List of K values to evaluate at

    Returns:
        Dictionary of metric_name -> value
    """
    if k_values is None:
        k_values = [5, 12, 20]

    results = defaultdict(list)
    all_recommended_items = set()

    evaluated_users = 0
    for user_id, actual in ground_truth.items():
        if not actual:
            continue
        predicted = predictions.get(user_id, [])
        if not predicted:
            # User got no recommendations — counts as 0 for all metrics
            for k in k_values:
                results[f"map@{k}"].append(0.0)
                results[f"recall@{k}"].append(0.0)
                results[f"precision@{k}"].append(0.0)
                results[f"ndcg@{k}"].append(0.0)
                results[f"hit_rate@{k}"].append(0.0)
            continue

        all_recommended_items.update(predicted)
        evaluated_users += 1

        for k in k_values:
            results[f"map@{k}"].append(average_precision_at_k(predicted, actual, k))
            results[f"recall@{k}"].append(recall_at_k(predicted, actual, k))
            results[f"precision@{k}"].append(precision_at_k(predicted, actual, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(predicted, actual, k))
            results[f"hit_rate@{k}"].append(hit_rate_at_k(predicted, actual, k))

    # Aggregate
    metrics = {name: float(np.mean(values)) for name, values in results.items()}

    # Catalog coverage: what fraction of items were recommended?
    all_items = set()
    for items in ground_truth.values():
        all_items.update(items)
    for items in predictions.values():
        all_items.update(items)
    if all_items:
        metrics["catalog_coverage"] = len(all_recommended_items) / len(all_items)

    metrics["evaluated_users"] = evaluated_users

    logger.info("Evaluation Results:")
    for name, value in sorted(metrics.items()):
        if name != "evaluated_users":
            logger.info(f"  {name}: {value:.4f}")

    return metrics


def compare_models(
    model_results: dict[str, dict[str, float]],
    primary_metric: str = "map@12",
) -> pd.DataFrame:
    """Compare multiple models side-by-side."""
    import pandas as pd

    rows = []
    for model_name, metrics in model_results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(primary_metric, ascending=False)
    return df
