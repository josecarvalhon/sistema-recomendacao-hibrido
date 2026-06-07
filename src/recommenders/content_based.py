"""Filtragem baseada em conteúdo via TF-IDF + atributos categóricos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.preprocessing import IndexMaps
from src.recommenders.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Representa cada material por um vetor textual+categórico e cada aluno por
    uma combinação do seu perfil cadastral com a média dos materiais consumidos.
    A pontuação é o cosseno entre o perfil do aluno e cada material."""

    name = "content"

    def __init__(self, max_features: int = 2000):
        self.max_features = max_features
        self.item_matrix: sparse.csr_matrix | None = None
        self.user_matrix: sparse.csr_matrix | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.cat_columns: list[str] = []
        self.cat_value_index: dict[tuple[str, str], int] = {}
        self.idx: IndexMaps | None = None

    @staticmethod
    def _item_text(materiais: pd.DataFrame) -> pd.Series:
        return (
            materiais["descricao"].fillna("") + " "
            + materiais["titulo"].fillna("") + " "
            + materiais["autor"].fillna("")
        ).str.lower()

    def _build_categorical(
        self,
        materiais: pd.DataFrame,
    ) -> sparse.csr_matrix:
        """One-hot esparso para tipo, area e nivel."""
        self.cat_columns = ["tipo", "area", "nivel"]
        self.cat_value_index = {}
        for col in self.cat_columns:
            for v in materiais[col].unique():
                self.cat_value_index.setdefault((col, str(v)), len(self.cat_value_index))
        n_cols = len(self.cat_value_index)
        rows, cols, vals = [], [], []
        for r_idx, row in enumerate(materiais.itertuples(index=False)):
            for col in self.cat_columns:
                key = (col, str(getattr(row, col)))
                c = self.cat_value_index[key]
                rows.append(r_idx)
                cols.append(c)
                vals.append(1.0)
        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(materiais), n_cols), dtype=np.float32
        )

    def _aluno_categorical(self, alunos: pd.DataFrame) -> sparse.csr_matrix:
        """Mapeia áreas de interesse do aluno para o mesmo espaço do material (col 'area')
        e o nível esperado a partir do período."""
        n_users = len(alunos)
        n_cols = len(self.cat_value_index)
        rows, cols, vals = [], [], []
        # Mapeamento período -> nível (regra simples, replicada no knowledge.py)
        def periodo_para_nivel(p: int) -> str:
            if p <= 3:
                return "Iniciante"
            if p <= 5:
                return "Intermediário"
            return "Avançado"

        for r_idx, row in enumerate(alunos.itertuples(index=False)):
            # Área de interesse
            for area in row.areas_lista:
                key = ("area", area)
                if key in self.cat_value_index:
                    rows.append(r_idx); cols.append(self.cat_value_index[key]); vals.append(1.0)
            # Nível esperado pelo período
            nivel = periodo_para_nivel(int(row.periodo))
            key = ("nivel", nivel)
            if key in self.cat_value_index:
                rows.append(r_idx); cols.append(self.cat_value_index[key]); vals.append(0.5)
        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_users, n_cols), dtype=np.float32
        )

    def fit(
        self,
        alunos: pd.DataFrame,
        materiais: pd.DataFrame,
        interacoes_train: pd.DataFrame,
        idx: IndexMaps,
    ) -> "ContentBasedRecommender":
        self.idx = idx

        # Garante mesma ordem do item_to_idx
        materiais_ord = materiais.set_index("id_material").loc[idx.idx_to_item].reset_index()
        alunos_ord = alunos.set_index("id_aluno").loc[idx.idx_to_user].reset_index()

        # Texto
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=None,
            min_df=2,
            ngram_range=(1, 2),
            lowercase=True,
        )
        text_matrix = self.vectorizer.fit_transform(self._item_text(materiais_ord))
        cat_matrix = self._build_categorical(materiais_ord)
        self.item_matrix = normalize(sparse.hstack([text_matrix, cat_matrix]).tocsr())

        # Perfil do aluno = perfil cadastral (cat) + média dos materiais consumidos
        user_cat = self._aluno_categorical(alunos_ord)
        # zero text part for cadastral (mesmo n_cols texto)
        zeros_text = sparse.csr_matrix((user_cat.shape[0], text_matrix.shape[1]), dtype=np.float32)
        user_cat_full = sparse.hstack([zeros_text, user_cat]).tocsr()

        # Histórico: média ponderada dos vetores dos itens consumidos (peso = avaliação)
        if len(interacoes_train) > 0:
            df = interacoes_train.copy()
            df["u"] = df["id_aluno"].map(idx.user_to_idx)
            df["i"] = df["id_material"].map(idx.item_to_idx)
            df = df.dropna(subset=["u", "i"])
            df["u"] = df["u"].astype(int); df["i"] = df["i"].astype(int)
            weights = df["avaliacao"].to_numpy(dtype=np.float32)
            interaction_csr = sparse.csr_matrix(
                (weights, (df["u"].to_numpy(), df["i"].to_numpy())),
                shape=(idx.n_users, idx.n_items),
                dtype=np.float32,
            )
            # normaliza por usuário
            sums = np.asarray(interaction_csr.sum(axis=1)).ravel()
            sums[sums == 0] = 1.0
            inv = sparse.diags(1.0 / sums)
            user_hist = inv @ interaction_csr @ self.item_matrix
        else:
            user_hist = sparse.csr_matrix(
                (idx.n_users, self.item_matrix.shape[1]), dtype=np.float32
            )

        # Combinação 0.5 cadastral + 0.5 histórico
        self.user_matrix = normalize(0.5 * user_cat_full + 0.5 * user_hist)
        return self

    def score_user(self, user_idx: int) -> np.ndarray:
        if self.user_matrix is None or self.item_matrix is None:
            raise RuntimeError("modelo não treinado")
        u_vec = self.user_matrix[user_idx]
        scores = u_vec @ self.item_matrix.T
        return np.asarray(scores.todense()).ravel()
