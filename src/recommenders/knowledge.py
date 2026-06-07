"""Recomendador baseado em conhecimento: regras simples sobre perfil acadêmico.

Regras:
- Materiais de área que coincide com 'areas_interesse' do aluno recebem +1.0
- Materiais de área que coincide com alguma 'disciplinas_cursadas' recebem +0.5
- Compatibilidade de nível com período acadêmico:
    período <= 3  -> Iniciante
    período 4-5   -> Intermediário
    período 6-8   -> Avançado
  Match de nível adiciona +0.7. Nível adjacente adiciona +0.3.
- Penalidade leve para nível desalinhado (-0.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import IndexMaps
from src.recommenders.base import BaseRecommender

NIVEL_RANK = {"Iniciante": 1, "Intermediário": 2, "Avançado": 3}

# Mapeamento das disciplinas/áreas de interesse para áreas dos materiais.
# As áreas dos materiais são: Programação, IA, Estatística, Gestão, Segurança, UX, BD, IoT.
DISCIPLINA_PARA_AREA = {
    "Algoritmos": "Programação",
    "Banco de Dados": "BD",
    "BD": "BD",
    "Estatística": "Estatística",
    "Python": "Programação",
    "ML": "IA",
    "IA": "IA",
    "Redes": "Segurança",
    "SO": "Programação",
    "Arquitetura": "Programação",
    "Gestão": "Gestão",
    "Web": "Programação",
    "Compiladores": "Programação",
    "UX": "UX",
    "PM": "Gestão",
}

INTERESSE_PARA_AREA = {
    "ML": "IA",
    "Programação": "Programação",
    "Cloud": "Programação",
    "IA": "IA",
    "Visualização": "Estatística",
    "Big Data": "Estatística",
    "IoT": "IoT",
    "Segurança": "Segurança",
    "Hardware": "IoT",
    "Embedded": "IoT",
    "Robótica": "IoT",
    "Deep Learning": "IA",
    "NLP": "IA",
    "QA": "Programação",
    "DevOps": "Programação",
    "Mobile": "Programação",
    "PM": "Gestão",
    "UX": "UX",
    "Analytics": "Estatística",
    "Marketing Digital": "Gestão",
}


def periodo_para_nivel(p: int) -> str:
    if p <= 3:
        return "Iniciante"
    if p <= 5:
        return "Intermediário"
    return "Avançado"


class KnowledgeRecommender(BaseRecommender):
    name = "knowledge"

    def __init__(self):
        self.idx: IndexMaps | None = None
        self.alunos: pd.DataFrame | None = None
        self.materiais: pd.DataFrame | None = None
        self._user_profiles: list[dict] = []

    def fit(
        self,
        alunos: pd.DataFrame,
        materiais: pd.DataFrame,
        idx: IndexMaps,
    ) -> "KnowledgeRecommender":
        self.idx = idx
        self.alunos = alunos.set_index("id_aluno").loc[idx.idx_to_user].reset_index()
        self.materiais = materiais.set_index("id_material").loc[idx.idx_to_item].reset_index()
        # pré-computa perfis
        self._user_profiles = []
        for row in self.alunos.itertuples(index=False):
            interesse_areas = {INTERESSE_PARA_AREA.get(a) for a in row.areas_lista}
            interesse_areas.discard(None)
            disc_areas = {DISCIPLINA_PARA_AREA.get(d) for d in row.disciplinas_lista}
            disc_areas.discard(None)
            nivel = periodo_para_nivel(int(row.periodo))
            self._user_profiles.append(
                {
                    "interesse_areas": interesse_areas,
                    "disc_areas": disc_areas,
                    "nivel": nivel,
                }
            )
        return self

    def score_user(self, user_idx: int) -> np.ndarray:
        if self.materiais is None:
            raise RuntimeError("modelo não treinado")
        prof = self._user_profiles[user_idx]
        scores = np.zeros(len(self.materiais), dtype=np.float32)
        nivel_alvo = NIVEL_RANK[prof["nivel"]]

        areas = self.materiais["area"].to_numpy()
        niveis = self.materiais["nivel"].to_numpy()

        # Áreas de interesse e disciplinas
        if prof["interesse_areas"]:
            scores += np.isin(areas, list(prof["interesse_areas"])).astype(np.float32) * 1.0
        if prof["disc_areas"]:
            scores += np.isin(areas, list(prof["disc_areas"])).astype(np.float32) * 0.5

        # Nível
        nivel_diff = np.abs(
            np.vectorize(NIVEL_RANK.get, otypes=[np.int8])(niveis) - nivel_alvo
        )
        scores += np.where(nivel_diff == 0, 0.7, 0.0).astype(np.float32)
        scores += np.where(nivel_diff == 1, 0.3, 0.0).astype(np.float32)
        scores += np.where(nivel_diff >= 2, -0.2, 0.0).astype(np.float32)

        return scores
