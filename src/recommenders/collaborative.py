"""Filtragem colaborativa via SVD truncada sobre a matriz aluno x material.

Decisão de projeto: para evitar dependências de bibliotecas pesadas (implicit,
surprise) e simplificar o Docker, usamos TruncatedSVD do scikit-learn sobre a
matriz centralizada de avaliações. Isso fornece tanto ranking (top-K) quanto
predição de avaliação (para RMSE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.preprocessing import IndexMaps
from src.recommenders.base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    name = "collab"

    def __init__(self, n_factors: int = 32, n_iter: int = 15, random_state: int = 42):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.random_state = random_state
        self.svd: TruncatedSVD | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.global_mean: float = 0.0
        self.idx: IndexMaps | None = None
        self.train_matrix: sparse.csr_matrix | None = None

    def fit(
        self,
        rating_matrix: sparse.csr_matrix,
        idx: IndexMaps,
    ) -> "CollaborativeRecommender":
        self.idx = idx
        self.train_matrix = rating_matrix.copy()

        nz = rating_matrix.nonzero()
        if len(nz[0]) == 0:
            raise ValueError("matriz de treino vazia")
        self.global_mean = float(rating_matrix.data.mean())

        # Centraliza apenas onde houver observação (mantendo a esparsidade)
        centered = rating_matrix.copy().astype(np.float32)
        centered.data = centered.data - self.global_mean

        n_factors = max(1, min(self.n_factors, min(centered.shape) - 1))
        self.svd = TruncatedSVD(
            n_components=n_factors,
            n_iter=self.n_iter,
            random_state=self.random_state,
            algorithm="randomized",
        )
        self.user_factors = self.svd.fit_transform(centered)
        # V tem shape (n_factors, n_items)
        self.item_factors = self.svd.components_.T  # (n_items, n_factors)
        return self

    def score_user(self, user_idx: int) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("modelo não treinado")
        return (self.user_factors[user_idx] @ self.item_factors.T) + self.global_mean

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("modelo não treinado")
        pred = float(self.user_factors[user_idx] @ self.item_factors[item_idx]) + self.global_mean
        return float(np.clip(pred, 1.0, 5.0))

    def predict_pairs(
        self, pairs: pd.DataFrame
    ) -> np.ndarray:
        """pairs com colunas u (user_idx) e i (item_idx)."""
        u = pairs["u"].to_numpy()
        i = pairs["i"].to_numpy()
        preds = np.einsum("ij,ij->i", self.user_factors[u], self.item_factors[i]) + self.global_mean
        return np.clip(preds, 1.0, 5.0)
