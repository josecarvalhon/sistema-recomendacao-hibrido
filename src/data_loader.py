"""Carregamento e filtragem dos três CSVs.

Aplica a estratégia A definida no projeto: trabalha apenas com o subset de
interações cuja dupla (id_aluno, id_material) está presente nos dois cadastros.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RawData:
    alunos: pd.DataFrame
    materiais: pd.DataFrame
    interacoes: pd.DataFrame
    integridade: dict


def _split_list_col(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).apply(
        lambda v: [x.strip() for x in v.split(",") if x.strip()]
    )


def load_raw(data_dir: Path) -> RawData:
    """Carrega os três CSVs e aplica o filtro de integridade referencial."""
    data_dir = Path(data_dir)
    logger.info("Carregando CSVs de %s", data_dir)

    alunos = pd.read_csv(data_dir / "dados_alunos.csv")
    materiais = pd.read_csv(data_dir / "materiais_didaticos.csv")
    interacoes = pd.read_csv(data_dir / "interacoes.csv")

    # Tipagem
    alunos["id_aluno"] = alunos["id_aluno"].astype(int)
    alunos["periodo"] = alunos["periodo"].astype(int)
    alunos["disciplinas_lista"] = _split_list_col(alunos["disciplinas_cursadas"])
    alunos["areas_lista"] = _split_list_col(alunos["areas_interesse"])

    materiais["id_material"] = materiais["id_material"].astype(int)

    interacoes["id_aluno"] = interacoes["id_aluno"].astype(int)
    interacoes["id_material"] = interacoes["id_material"].astype(int)
    interacoes["avaliacao"] = interacoes["avaliacao"].astype(int)
    interacoes["duracao_minutos"] = interacoes["duracao_minutos"].astype(int)
    interacoes["data"] = pd.to_datetime(interacoes["data"])

    total_inter = len(interacoes)

    # Filtro estrutural: apenas IDs presentes nos dois cadastros (estratégia A)
    ids_alunos = set(alunos["id_aluno"])
    ids_mat = set(materiais["id_material"])
    mask = interacoes["id_aluno"].isin(ids_alunos) & interacoes["id_material"].isin(ids_mat)
    interacoes_filt = interacoes.loc[mask].copy()

    integridade = {
        "interacoes_originais": int(total_inter),
        "interacoes_apos_filtro": int(len(interacoes_filt)),
        "fracao_aproveitada": round(len(interacoes_filt) / total_inter, 6),
        "alunos_cadastrados": int(len(ids_alunos)),
        "materiais_cadastrados": int(len(ids_mat)),
        "alunos_com_interacao_apos_filtro": int(interacoes_filt["id_aluno"].nunique()),
        "materiais_com_interacao_apos_filtro": int(interacoes_filt["id_material"].nunique()),
    }
    logger.info(
        "Subset cruzável: %s/%s interações (%.2f%%)",
        integridade["interacoes_apos_filtro"],
        integridade["interacoes_originais"],
        integridade["fracao_aproveitada"] * 100,
    )

    # Remove duplicatas exatas no subset (raras)
    before = len(interacoes_filt)
    interacoes_filt = interacoes_filt.drop_duplicates(
        subset=["id_aluno", "id_material", "tipo_interacao", "data"]
    )
    integridade["duplicatas_removidas"] = int(before - len(interacoes_filt))

    return RawData(
        alunos=alunos,
        materiais=materiais,
        interacoes=interacoes_filt,
        integridade=integridade,
    )


def filter_min_counts(
    interacoes: pd.DataFrame,
    min_user: int = 5,
    min_item: int = 5,
    max_iters: int = 5,
) -> pd.DataFrame:
    """Aplica iterativamente um corte mínimo de interações por usuário e item."""
    df = interacoes.copy()
    for _ in range(max_iters):
        u_counts = df["id_aluno"].value_counts()
        i_counts = df["id_material"].value_counts()
        keep_u = u_counts[u_counts >= min_user].index
        keep_i = i_counts[i_counts >= min_item].index
        new_df = df[df["id_aluno"].isin(keep_u) & df["id_material"].isin(keep_i)]
        if len(new_df) == len(df):
            break
        df = new_df
    return df.reset_index(drop=True)
