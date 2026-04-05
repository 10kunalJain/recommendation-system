"""Two-Tower Neural Retrieval Model.

Architecture used by YouTube, Google, and Meta for large-scale retrieval.
Learns separate embeddings for users and items, scores via dot product.

Why Two-Tower?
- Decoupled towers allow precomputing item embeddings → fast online inference
- Scales to millions of items (ANN search on precomputed embeddings)
- Captures non-linear feature interactions that ALS misses

This model is trained on implicit feedback (purchase = positive, random = negative).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger


class UserTower(nn.Module):
    """Encodes user features into a dense embedding."""

    def __init__(self, n_users: int, embedding_dim: int = 64, n_features: int = 5):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_dim),
        )
        self.combine = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        emb = self.user_embedding(user_ids)
        feat = self.feature_mlp(user_features)
        combined = torch.cat([emb, feat], dim=-1)
        return F.normalize(self.combine(combined), p=2, dim=-1)


class ItemTower(nn.Module):
    """Encodes item features into a dense embedding."""

    def __init__(self, n_items: int, embedding_dim: int = 64, n_features: int = 4):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, embedding_dim),
        )
        self.combine = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(item_ids)
        feat = self.feature_mlp(item_features)
        combined = torch.cat([emb, feat], dim=-1)
        return F.normalize(self.combine(combined), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Two-tower retrieval model with in-batch negative sampling."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_user_features: int = 5,
        n_item_features: int = 4,
    ):
        super().__init__()
        self.user_tower = UserTower(n_users, embedding_dim, n_user_features)
        self.item_tower = ItemTower(n_items, embedding_dim, n_item_features)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        # Cosine similarity scaled by temperature
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature.abs()
        return logits

    def get_user_embedding(
        self, user_ids: torch.Tensor, user_features: torch.Tensor
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            emb = self.user_tower(user_ids, user_features)
        return emb.cpu().numpy()

    def get_item_embedding(
        self, item_ids: torch.Tensor, item_features: torch.Tensor
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            emb = self.item_tower(item_ids, item_features)
        return emb.cpu().numpy()


class TwoTowerTrainer:
    """Training loop with in-batch negative sampling."""

    def __init__(
        self,
        model: TwoTowerModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            user_ids, user_feats, item_ids, item_feats = batch

            # In-batch negatives: each positive pair is a negative for others
            logits = self.model(user_ids, user_feats, item_ids, item_feats)
            labels = torch.arange(len(user_ids))  # diagonal = positive pairs

            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, dataloader, n_epochs: int = 10):
        logger.info(f"Training Two-Tower model for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            loss = self.train_epoch(dataloader)
            if (epoch + 1) % 2 == 0:
                logger.info(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")
        logger.info("Two-Tower training complete.")
