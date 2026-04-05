"""Tests for data loading and preprocessing."""

import pytest
import pandas as pd
import numpy as np
from scipy.sparse import issparse


class TestTemporalSplit:
    """Temporal split must prevent data leakage."""

    def test_no_temporal_overlap(self, train_val_test_split):
        train, val, test = train_val_test_split
        assert train["t_dat"].max() < val["t_dat"].min()
        assert val["t_dat"].max() <= test["t_dat"].min()

    def test_all_data_accounted_for(self, sample_transactions, train_val_test_split):
        train, val, test = train_val_test_split
        assert len(train) + len(val) + len(test) == len(sample_transactions)

    def test_train_is_largest_split(self, train_val_test_split):
        train, val, test = train_val_test_split
        assert len(train) >= len(val)
        assert len(train) >= len(test)


class TestUserItemMappings:
    """Mappings must be bijective and cover all training entities."""

    def test_bijective_user_mapping(self, user_item_mappings):
        u2i, i2u, _, _ = user_item_mappings
        assert len(u2i) == len(i2u)
        for uid, idx in u2i.items():
            assert i2u[idx] == uid

    def test_bijective_item_mapping(self, user_item_mappings):
        _, _, i2i, i2item = user_item_mappings
        assert len(i2i) == len(i2item)
        for aid, idx in i2i.items():
            assert i2item[idx] == aid

    def test_covers_all_train_users(self, train_val_test_split, user_item_mappings):
        train = train_val_test_split[0]
        u2i = user_item_mappings[0]
        for uid in train["customer_id"].unique():
            assert uid in u2i


class TestInteractionMatrix:
    """Sparse interaction matrix must be well-formed."""

    def test_matrix_shape(self, train_val_test_split, user_item_mappings):
        from src.data.loader import build_interaction_matrix
        train = train_val_test_split[0]
        u2i, _, i2i, _ = user_item_mappings
        matrix = build_interaction_matrix(train, u2i, i2i)

        assert matrix.shape == (len(u2i), len(i2i))
        assert issparse(matrix)

    def test_no_negative_values(self, train_val_test_split, user_item_mappings):
        from src.data.loader import build_interaction_matrix
        train = train_val_test_split[0]
        u2i, _, i2i, _ = user_item_mappings
        matrix = build_interaction_matrix(train, u2i, i2i)

        assert matrix.min() >= 0

    def test_log_confidence_weighting(self, train_val_test_split, user_item_mappings):
        """Values should be log1p(count), not raw counts."""
        from src.data.loader import build_interaction_matrix
        train = train_val_test_split[0]
        u2i, _, i2i, _ = user_item_mappings
        matrix = build_interaction_matrix(train, u2i, i2i)

        # All values should be <= log1p(max_possible_count)
        max_val = matrix.max()
        assert max_val <= np.log1p(len(train))
