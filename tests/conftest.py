"""Fixtures pequenas e independentes do dataset real."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import _split_list_col


@pytest.fixture
def alunos_mini() -> pd.DataFrame:
    df = pd.DataFrame({
        "id_aluno": [1, 2, 3],
        "nome": ["A", "B", "C"],
        "curso": ["Ciência de Dados", "Engenharia de Software", "SI"],
        "periodo": [3, 5, 7],
        "disciplinas_cursadas": ["Python, ML", "Algoritmos, BD", "Redes, SO"],
        "areas_interesse": ["IA, ML", "Programação, Cloud", "Segurança, IoT"],
    })
    df["disciplinas_lista"] = _split_list_col(df["disciplinas_cursadas"])
    df["areas_lista"] = _split_list_col(df["areas_interesse"])
    return df


@pytest.fixture
def materiais_mini() -> pd.DataFrame:
    return pd.DataFrame({
        "id_material": [10, 20, 30, 40, 50],
        "titulo": ["Intro ML", "Algoritmos I", "Banco de Dados", "Segurança em Redes", "Python Básico"],
        "tipo":   ["livro", "livro", "artigo", "vídeo", "livro"],
        "area":   ["IA", "Programação", "BD", "Segurança", "Programação"],
        "nivel":  ["Intermediário", "Iniciante", "Intermediário", "Avançado", "Iniciante"],
        "descricao": [
            "introducao a machine learning supervisionado",
            "estrutura de dados e algoritmos basicos",
            "modelagem de dados e SQL",
            "principios de seguranca em redes corporativas",
            "fundamentos de python para iniciantes",
        ],
        "autor": ["X", "Y", "Z", "W", "K"],
    })


@pytest.fixture
def interacoes_mini() -> pd.DataFrame:
    return pd.DataFrame({
        "id_interacao": list(range(1, 11)),
        "id_aluno":     [1, 1, 2, 2, 2, 3, 3, 1, 2, 3],
        "id_material":  [10, 20, 20, 30, 50, 40, 50, 50, 40, 30],
        "tipo_interacao": ["leitura"] * 10,
        "avaliacao":      [5, 4, 5, 4, 3, 5, 4, 4, 3, 4],
        "duracao_minutos": [60] * 10,
        "data":           pd.to_datetime(["2023-01-01"] * 10),
    })
