"""API REST do sistema de recomendação híbrido.

Endpoints:
    GET /health
    GET /alunos/{id_aluno}
    GET /materiais/{id_material}
    GET /alunos/{id_aluno}/historico
    GET /recomendacoes/{id_aluno}?k=10&strategy=hybrid|content|collab|knowledge
    GET /metrics
    GET /info
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import settings
from src.recommenders.hybrid import HybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

# Estado carregado em startup
_state: dict = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if settings.artifact_path.exists():
        logger.info("Carregando bundle %s...", settings.artifact_path)
        bundle = joblib.load(settings.artifact_path)
        _state["bundle"] = bundle
        _state["hybrid"] = HybridRecommender(
            bundle["content"], bundle["collab"], bundle["knowledge"],
            weights=tuple(bundle["hybrid_weights"]),
        )
        logger.info("Pronto: %d alunos x %d materiais",
                    bundle["idx"].n_users, bundle["idx"].n_items)
    else:
        logger.warning("Artefato não encontrado em %s — endpoints de recomendação retornarão 503",
                       settings.artifact_path)
    yield


app = FastAPI(
    title="Sistema de Recomendação Híbrido — Plataforma de Ensino",
    description=(
        "API REST para recomendação de materiais didáticos (livros, artigos, vídeos) "
        "com base em filtragem por conteúdo, colaborativa e conhecimento."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


# --------------------------- Modelos Pydantic ---------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    artifact_loaded: bool
    n_users: int = 0
    n_items: int = 0


class Aluno(BaseModel):
    id_aluno: int
    nome: str
    curso: str
    periodo: int
    disciplinas_cursadas: list[str]
    areas_interesse: list[str]


class Material(BaseModel):
    id_material: int
    titulo: str
    tipo: str
    area: str
    nivel: str
    descricao: str
    autor: str


class Recomendacao(BaseModel):
    id_material: int
    titulo: str
    tipo: str
    area: str
    nivel: str
    autor: str
    score: float = Field(..., description="Score normalizado retornado pelo modelo")


class RecommendResponse(BaseModel):
    id_aluno: int
    strategy: str
    k: int
    items: list[Recomendacao]


class HistoricoItem(BaseModel):
    id_material: int
    titulo: str | None = None
    avaliacao_media: float
    n_interacoes: int


# --------------------------- Helpers ---------------------------

def _require_bundle() -> dict:
    if "bundle" not in _state:
        raise HTTPException(status_code=503, detail="Modelos ainda não treinados. Execute o serviço 'trainer'.")
    return _state["bundle"]


def _aluno_row(bundle: dict, id_aluno: int):
    df = bundle["alunos"]
    sub = df[df["id_aluno"] == id_aluno]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Aluno {id_aluno} não encontrado")
    return sub.iloc[0]


def _material_row(bundle: dict, id_material: int):
    df = bundle["materiais"]
    sub = df[df["id_material"] == id_material]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Material {id_material} não encontrado")
    return sub.iloc[0]


# --------------------------- Endpoints ---------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if "bundle" not in _state:
        return HealthResponse(status="degraded", artifact_loaded=False)
    idx = _state["bundle"]["idx"]
    return HealthResponse(
        status="ok", artifact_loaded=True,
        n_users=idx.n_users, n_items=idx.n_items,
    )


@app.get("/info")
def info() -> dict:
    bundle = _require_bundle()
    return {
        "n_users": bundle["idx"].n_users,
        "n_items": bundle["idx"].n_items,
        "default_top_k": bundle["top_k"],
        "hybrid_weights": bundle["hybrid_weights"],
        "strategies": ["hybrid", "content", "collab", "knowledge"],
    }


@app.get("/metrics")
def metrics() -> dict:
    if not settings.metrics_path.exists():
        raise HTTPException(status_code=503, detail="Métricas indisponíveis (treine os modelos).")
    return json.loads(settings.metrics_path.read_text())


@app.get("/alunos/{id_aluno}", response_model=Aluno)
def get_aluno(id_aluno: int) -> Aluno:
    bundle = _require_bundle()
    row = _aluno_row(bundle, id_aluno)
    return Aluno(
        id_aluno=int(row["id_aluno"]),
        nome=str(row["nome"]),
        curso=str(row["curso"]),
        periodo=int(row["periodo"]),
        disciplinas_cursadas=list(row["disciplinas_lista"]),
        areas_interesse=list(row["areas_lista"]),
    )


@app.get("/materiais/{id_material}", response_model=Material)
def get_material(id_material: int) -> Material:
    bundle = _require_bundle()
    row = _material_row(bundle, id_material)
    return Material(
        id_material=int(row["id_material"]),
        titulo=str(row["titulo"]),
        tipo=str(row["tipo"]),
        area=str(row["area"]),
        nivel=str(row["nivel"]),
        descricao=str(row["descricao"]),
        autor=str(row["autor"]),
    )


@app.get("/alunos/{id_aluno}/historico", response_model=list[HistoricoItem])
def get_historico(id_aluno: int, limit: int = Query(20, ge=1, le=200)) -> list[HistoricoItem]:
    bundle = _require_bundle()
    idx = bundle["idx"]
    user_idx = idx.user_to_idx.get(int(id_aluno))
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"Aluno {id_aluno} não está no subset cadastrado")
    hist = bundle.get("historico_df")
    if hist is None or hist.empty:
        return []
    sub = hist[hist["id_aluno"] == id_aluno].head(limit)
    materiais = bundle["materiais"].set_index("id_material")
    out: list[HistoricoItem] = []
    for row in sub.itertuples(index=False):
        mid = int(row.id_material)
        titulo = str(materiais.loc[mid, "titulo"]) if mid in materiais.index else None
        out.append(HistoricoItem(
            id_material=mid,
            titulo=titulo,
            avaliacao_media=float(row.avaliacao_media),
            n_interacoes=int(row.n_interacoes),
        ))
    return out


@app.get("/recomendacoes/{id_aluno}", response_model=RecommendResponse)
def recomendar(
    id_aluno: int,
    k: int = Query(10, ge=1, le=100),
    strategy: Literal["hybrid", "content", "collab", "knowledge"] = "hybrid",
    excluir_consumidos: bool = True,
) -> RecommendResponse:
    bundle = _require_bundle()
    idx = bundle["idx"]
    user_idx = idx.user_to_idx.get(int(id_aluno))
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"Aluno {id_aluno} não está no subset cadastrado")

    model_map = {
        "hybrid": _state["hybrid"],
        "content": bundle["content"],
        "collab": bundle["collab"],
        "knowledge": bundle["knowledge"],
    }
    model = model_map[strategy]

    excl = bundle["train_dict"].get(user_idx, set()) if excluir_consumidos else set()
    rec_idx, rec_scores = model.recommend(user_idx, k=k, exclude_items=excl)

    materiais_df = bundle["materiais"].set_index("id_material")
    items: list[Recomendacao] = []
    for ii, sc in zip(rec_idx, rec_scores):
        mid = int(idx.idx_to_item[int(ii)])
        if mid not in materiais_df.index:
            continue
        r = materiais_df.loc[mid]
        items.append(Recomendacao(
            id_material=mid,
            titulo=str(r["titulo"]),
            tipo=str(r["tipo"]),
            area=str(r["area"]),
            nivel=str(r["nivel"]),
            autor=str(r["autor"]),
            score=float(sc) if np.isfinite(sc) else 0.0,
        ))
    return RecommendResponse(id_aluno=id_aluno, strategy=strategy, k=k, items=items)
