"""Interface comum dos recomendadores."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRecommender(ABC):
    """Recomendador retorna um vetor de scores sobre todos os itens, indexado por idx_item."""

    name: str = "base"

    @abstractmethod
    def fit(self, **kwargs) -> "BaseRecommender":
        ...

    @abstractmethod
    def score_user(self, user_idx: int) -> np.ndarray:
        """Retorna scores (n_items,) para o usuário."""

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_items: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (top_k_idx, top_k_scores) ordenados por score decrescente."""
        scores = self.score_user(user_idx).astype(float).copy()
        if exclude_items:
            scores[list(exclude_items)] = -np.inf
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return order, scores[order]
