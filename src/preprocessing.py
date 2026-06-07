"""Construção de matrizes esparsas e splits para treino e avaliação."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


@dataclass
class IndexMaps:
    user_to_idx: dict[int, int]
    item_to_idx: dict[int, int]
    idx_to_user: np.ndarray
    idx_to_item: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def n_items(self) -> int:
        return len(self.item_to_idx)


def build_index_maps(
    alunos: pd.DataFrame, materiais: pd.DataFrame
) -> IndexMaps:
    user_ids = np.sort(alunos["id_aluno"].unique())
    item_ids = np.sort(materiais["id_material"].unique())
    return IndexMaps(
        user_to_idx={int(u): i for i, u in enumerate(user_ids)},
        item_to_idx={int(m): i for i, m in enumerate(item_ids)},
        idx_to_user=user_ids,
        idx_to_item=item_ids,
    )


def split_train_test(
    interacoes: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split por usuário: para cada usuário com >= 2 interações, separa uma fração para teste."""
    rng = np.random.default_rng(random_state)
    test_idx = []
    grouped = interacoes.groupby("id_aluno", sort=False).indices
    for user, idxs in grouped.items():
        n = len(idxs)
        if n < 2:
            continue
        n_test = max(1, int(round(n * test_fraction)))
        chosen = rng.choice(idxs, size=n_test, replace=False)
        test_idx.extend(chosen.tolist())
    test_idx = np.array(sorted(test_idx))
    test_mask = np.zeros(len(interacoes), dtype=bool)
    test_mask[test_idx] = True
    test = interacoes.iloc[test_mask].reset_index(drop=True)
    train = interacoes.iloc[~test_mask].reset_index(drop=True)
    logger.info("Split: train=%d, test=%d", len(train), len(test))
    return train, test


def build_rating_matrix(
    interacoes: pd.DataFrame, idx: IndexMaps
) -> sparse.csr_matrix:
    """Matriz aluno x material com a média da avaliação por par."""
    df = interacoes.groupby(["id_aluno", "id_material"], as_index=False)["avaliacao"].mean()
    rows = df["id_aluno"].map(idx.user_to_idx).to_numpy()
    cols = df["id_material"].map(idx.item_to_idx).to_numpy()
    vals = df["avaliacao"].to_numpy(dtype=np.float32)
    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(idx.n_users, idx.n_items)
    )


def build_implicit_matrix(
    interacoes: pd.DataFrame, idx: IndexMaps
) -> sparse.csr_matrix:
    """Matriz binária aluno x material indicando interação."""
    df = interacoes[["id_aluno", "id_material"]].drop_duplicates()
    rows = df["id_aluno"].map(idx.user_to_idx).to_numpy()
    cols = df["id_material"].map(idx.item_to_idx).to_numpy()
    vals = np.ones(len(df), dtype=np.float32)
    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(idx.n_users, idx.n_items)
    )
