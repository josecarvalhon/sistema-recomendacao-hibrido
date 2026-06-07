"""Combinador híbrido: pondera scores normalizados dos três recomendadores."""
from __future__ import annotations

import numpy as np

from src.recommenders.base import BaseRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.knowledge import KnowledgeRecommender


def _minmax_norm(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v)
    lo = v[finite].min()
    hi = v[finite].max()
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    out = (v - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(
        self,
        content: ContentBasedRecommender,
        collab: CollaborativeRecommender,
        knowledge: KnowledgeRecommender,
        weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    ):
        self.content = content
        self.collab = collab
        self.knowledge = knowledge
        w = np.array(weights, dtype=float)
        if w.sum() <= 0:
            raise ValueError("soma dos pesos deve ser > 0")
        self.weights = w / w.sum()

    def fit(self, **_) -> "HybridRecommender":
        # Os componentes já chegam treinados
        return self

    def score_user(self, user_idx: int) -> np.ndarray:
        s_content = _minmax_norm(self.content.score_user(user_idx))
        s_collab = _minmax_norm(self.collab.score_user(user_idx))
        s_knowledge = _minmax_norm(self.knowledge.score_user(user_idx))
        w_c, w_co, w_k = self.weights
        return w_c * s_content + w_co * s_collab + w_k * s_knowledge
