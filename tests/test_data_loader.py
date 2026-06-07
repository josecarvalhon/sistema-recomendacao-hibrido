from __future__ import annotations

import pandas as pd

from src.data_loader import filter_min_counts


def test_filter_min_counts_remove_baixa_freq():
    df = pd.DataFrame({
        "id_aluno":     [1, 1, 1, 1, 1, 2, 2, 2, 3],
        "id_material":  [10, 11, 12, 13, 14, 10, 11, 12, 99],
    })
    out = filter_min_counts(df, min_user=3, min_item=2)
    # aluno 3 só tem 1 interação -> remove; material 99 só tem 1 interação -> remove
    assert 3 not in out["id_aluno"].unique()
    assert 99 not in out["id_material"].unique()


def test_filter_min_counts_idempotente_quando_atende():
    df = pd.DataFrame({
        "id_aluno":     [1, 1, 1, 2, 2, 2],
        "id_material":  [10, 20, 30, 10, 20, 30],
    })
    out = filter_min_counts(df, min_user=3, min_item=2)
    assert len(out) == len(df)
