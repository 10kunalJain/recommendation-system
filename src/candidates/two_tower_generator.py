"""Two-Tower Neural Candidate Generator.

Wraps the Two-Tower model for use in the candidate generation pipeline.
Trains on (user, item) positive pairs with in-batch negatives, then
uses precomputed item embeddings + ANN search for fast retrieval.

This captures non-linear feature interactions that ALS misses —
e.g., "young users who buy basics also like streetwear" requires
crossing age with product type, which linear models can't express.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from src.models.two_tower import TwoTowerModel, TwoTowerTrainer


class InteractionDataset(Dataset):
    """Dataset of (user, item) positive pairs with features."""

    def __init__(
        self,
        user_ids: np.ndarray,
        user_features: np.ndarray,
        item_ids: np.ndarray,
        item_features: np.ndarray,
    ):
        self.user_ids = torch.LongTensor(user_ids)
        self.user_features = torch.FloatTensor(user_features)
        self.item_ids = torch.LongTensor(item_ids)
        self.item_features = torch.FloatTensor(item_features)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.user_features[idx],
            self.item_ids[idx],
            self.item_features[idx],
        )


class TwoTowerCandidateGenerator:
    """Candidate generator using Two-Tower neural retrieval."""

    USER_FEATURE_COLS = [
        "purchase_count", "unique_items", "purchase_recency_days",
        "purchase_frequency", "color_diversity",
    ]
    ITEM_FEATURE_COLS = [
        "total_purchases", "unique_buyers", "repurchase_rate",
        "days_since_first_purchase",
    ]

    def __init__(self, embedding_dim: int = 64, n_epochs: int = 10, batch_size: int = 512):
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model: TwoTowerModel | None = None
        self.item_embeddings: np.ndarray | None = None
        self.item_ids_ordered: list[str] = []

    def fit(
        self,
        transactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
    ):
        """Train Two-Tower model on interaction data."""
        logger.info("Training Two-Tower neural retrieval model...")

        n_users = len(user_to_idx)
        n_items = len(item_to_idx)

        # Build user feature lookup (idx -> feature vector)
        user_feat_lookup = self._build_feature_lookup(
            user_features, "customer_id", user_to_idx, self.USER_FEATURE_COLS
        )
        item_feat_lookup = self._build_feature_lookup(
            item_features, "article_id", item_to_idx, self.ITEM_FEATURE_COLS
        )

        # Build training pairs from transactions
        user_ids, user_feats, item_ids, item_feats = [], [], [], []
        for _, row in transactions.iterrows():
            uid = row["customer_id"]
            iid = row["article_id"]
            if uid in user_to_idx and iid in item_to_idx:
                uidx = user_to_idx[uid]
                iidx = item_to_idx[iid]
                user_ids.append(uidx)
                item_ids.append(iidx)
                user_feats.append(user_feat_lookup.get(uidx, np.zeros(len(self.USER_FEATURE_COLS))))
                item_feats.append(item_feat_lookup.get(iidx, np.zeros(len(self.ITEM_FEATURE_COLS))))

        dataset = InteractionDataset(
            np.array(user_ids), np.array(user_feats),
            np.array(item_ids), np.array(item_feats),
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            n_user_features=len(self.USER_FEATURE_COLS),
            n_item_features=len(self.ITEM_FEATURE_COLS),
        )

        trainer = TwoTowerTrainer(self.model)
        trainer.train(dataloader, n_epochs=self.n_epochs)

        # Precompute all item embeddings for fast retrieval
        self._precompute_item_embeddings(item_to_idx, item_feat_lookup)
        logger.info("Two-Tower candidate generator ready.")

    def _build_feature_lookup(
        self, features_df: pd.DataFrame, id_col: str,
        id_to_idx: dict, feature_cols: list[str],
    ) -> dict[int, np.ndarray]:
        """Build idx -> feature vector mapping."""
        lookup = {}
        available_cols = [c for c in feature_cols if c in features_df.columns]
        for _, row in features_df.iterrows():
            entity_id = row[id_col]
            if entity_id in id_to_idx:
                idx = id_to_idx[entity_id]
                feats = [float(row.get(c, 0) or 0) for c in available_cols]
                # Pad if fewer features available
                feats += [0.0] * (len(feature_cols) - len(feats))
                lookup[idx] = np.array(feats, dtype=np.float32)
        return lookup

    def _precompute_item_embeddings(self, item_to_idx: dict, item_feat_lookup: dict):
        """Precompute item tower embeddings for all items."""
        self.model.eval()
        n_items = len(item_to_idx)
        idx_to_item = {v: k for k, v in item_to_idx.items()}
        self.item_ids_ordered = [idx_to_item[i] for i in range(n_items)]

        all_item_ids = torch.arange(n_items)
        all_item_feats = torch.FloatTensor([
            item_feat_lookup.get(i, np.zeros(len(self.ITEM_FEATURE_COLS)))
            for i in range(n_items)
        ])

        with torch.no_grad():
            self.item_embeddings = (
                self.model.item_tower(all_item_ids, all_item_feats).cpu().numpy()
            )
        logger.info(f"Precomputed {n_items} item embeddings ({self.embedding_dim}d)")

    def generate_candidates(
        self,
        user_idx: int,
        user_features: np.ndarray,
        n_candidates: int = 50,
        exclude_items: set[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top-N candidates via dot-product similarity."""
        if self.model is None or self.item_embeddings is None:
            return []

        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.user_tower(
                torch.LongTensor([user_idx]),
                torch.FloatTensor([user_features]),
            ).cpu().numpy()[0]

        # Dot product with all item embeddings
        scores = self.item_embeddings @ user_emb
        top_indices = np.argsort(scores)[::-1]

        candidates = []
        for idx in top_indices:
            if exclude_items and idx in exclude_items:
                continue
            candidates.append((self.item_ids_ordered[idx], float(scores[idx])))
            if len(candidates) >= n_candidates:
                break

        return candidates
