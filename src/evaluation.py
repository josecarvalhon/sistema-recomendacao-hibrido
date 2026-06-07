"""Métricas offline para os recomendadores.

- Precision@K, Recall@K, F1@K (baseado em interações observadas em test)
- RMSE (apenas para o colaborativo, sobre avaliações observadas em test)
- Coverage (fração de itens do catálogo que aparecem em alguma top-K)
- Diversity (1 - cosseno médio entre itens dentro da top-K, agregado por usuário)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import IndexMaps
from src.recommenders.base import BaseRecommender
from src.recommenders.collaborative import CollaborativeRecommender

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    model: str
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    coverage: float
    diversity: float
    rmse: float | None
    k: int
    n_users_evaluated: int

    def to_dict(self) -> dict:
        return asdict(self)


def _build_test_dict(test: pd.DataFrame, idx: IndexMaps) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for u, i in zip(test["id_aluno"].to_numpy(), test["id_material"].to_numpy()):
        ui = idx.user_to_idx.get(int(u))
        ii = idx.item_to_idx.get(int(i))
        if ui is None or ii is None:
            continue
        out.setdefault(ui, set()).add(ii)
    return out


def _build_train_dict(train: pd.DataFrame, idx: IndexMaps) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for u, i in zip(train["id_aluno"].to_numpy(), train["id_material"].to_numpy()):
        ui = idx.user_to_idx.get(int(u))
        ii = idx.item_to_idx.get(int(i))
        if ui is None or ii is None:
            continue
        out.setdefault(ui, set()).add(ii)
    return out


def _diversity_for_recs(
    rec_lists: list[np.ndarray],
    item_features: sparse.csr_matrix,
) -> float:
    """Diversidade média intra-lista = 1 - média do cosseno par-a-par."""
    if not rec_lists:
        return 0.0
    vals = []
    for rec in rec_lists:
        if len(rec) < 2:
            continue
        vecs = item_features[rec]
        sims = cosine_similarity(vecs)
        n = sims.shape[0]
        # média do triangular superior, sem a diagonal
        iu = np.triu_indices(n, k=1)
        vals.append(1.0 - float(sims[iu].mean()))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_topk(
    model: BaseRecommender,
    train_dict: dict[int, set[int]],
    test_dict: dict[int, set[int]],
    n_items: int,
    k: int,
    item_features: sparse.csr_matrix | None = None,
    sample_users: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Avalia ranking top-K do modelo nos usuários presentes em test_dict."""
    users = sorted(test_dict.keys())
    if sample_users is not None and sample_users < len(users):
        rng = rng or np.random.default_rng(42)
        users = list(rng.choice(users, size=sample_users, replace=False))

    precs, recs = [], []
    catalog_hits: set[int] = set()
    rec_lists: list[np.ndarray] = []

    for u in users:
        excl = train_dict.get(u, set())
        rec_idx, _ = model.recommend(u, k=k, exclude_items=excl)
        relevant = test_dict.get(u, set())
        if not relevant:
            continue
        rec_set = set(int(x) for x in rec_idx)
        hits = len(rec_set & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
        catalog_hits.update(rec_set)
        rec_lists.append(rec_idx)

    precision = float(np.mean(precs)) if precs else 0.0
    recall = float(np.mean(recs)) if recs else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )
    coverage = len(catalog_hits) / n_items if n_items > 0 else 0.0
    diversity = (
        _diversity_for_recs(rec_lists, item_features)
        if item_features is not None
        else 0.0
    )

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "coverage": coverage,
        "diversity": diversity,
        "n_users_evaluated": len(precs),
    }


def evaluate_rmse(
    model: CollaborativeRecommender,
    test: pd.DataFrame,
    idx: IndexMaps,
) -> float:
    df = test.copy()
    df["u"] = df["id_aluno"].map(idx.user_to_idx)
    df["i"] = df["id_material"].map(idx.item_to_idx)
    df = df.dropna(subset=["u", "i"]).copy()
    if df.empty:
        return float("nan")
    df["u"] = df["u"].astype(int); df["i"] = df["i"].astype(int)
    preds = model.predict_pairs(df[["u", "i"]])
    actual = df["avaliacao"].to_numpy(dtype=np.float32)
    return float(np.sqrt(np.mean((preds - actual) ** 2)))
