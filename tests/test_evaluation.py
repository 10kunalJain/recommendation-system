"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    evaluate_recommendations,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2/3)

    def test_empty_predictions(self):
        assert precision_at_k([], {"a"}, 5) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, 5) == 0.5

    def test_empty_actual(self):
        assert recall_at_k(["a"], set(), 5) == 0.0


class TestAveragePrecisionAtK:
    def test_perfect_ap(self):
        result = average_precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3)
        assert result == pytest.approx(1.0)

    def test_position_matters(self):
        # Relevant item at rank 1 should have higher AP than at rank 3
        ap_early = average_precision_at_k(["a", "x", "x"], {"a"}, 3)
        ap_late = average_precision_at_k(["x", "x", "a"], {"a"}, 3)
        assert ap_early > ap_late

    def test_empty_returns_zero(self):
        assert average_precision_at_k([], {"a"}, 5) == 0.0


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_position_discount(self):
        # Relevant item at position 1 should give higher NDCG
        ndcg_first = ndcg_at_k(["a", "x"], {"a"}, 2)
        ndcg_second = ndcg_at_k(["x", "a"], {"a"}, 2)
        assert ndcg_first > ndcg_second


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["x", "y"], {"a"}, 2) == 0.0


class TestEvaluateRecommendations:
    def test_returns_all_metrics(self):
        predictions = {"u1": ["a", "b", "c"]}
        ground_truth = {"u1": {"a", "d"}}

        metrics = evaluate_recommendations(predictions, ground_truth, [3])
        assert "map@3" in metrics
        assert "recall@3" in metrics
        assert "precision@3" in metrics
        assert "ndcg@3" in metrics
        assert "hit_rate@3" in metrics
        assert "catalog_coverage" in metrics

    def test_empty_predictions_score_zero(self):
        predictions = {"u1": []}
        ground_truth = {"u1": {"a"}}

        metrics = evaluate_recommendations(predictions, ground_truth, [5])
        assert metrics["map@5"] == 0.0
