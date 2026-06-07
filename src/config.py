"""Configuração central do sistema, lida via variáveis de ambiente."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _env_path("DATA_DIR", "/app/data")
    artifacts_dir: Path = _env_path("ARTIFACTS_DIR", "/app/artifacts")
    random_seed: int = _env_int("RANDOM_SEED", 42)
    test_fraction: float = _env_float("TEST_FRACTION", 0.2)
    min_interactions_user: int = _env_int("MIN_INTERACTIONS_USER", 5)
    min_interactions_item: int = _env_int("MIN_INTERACTIONS_ITEM", 5)
    top_k: int = _env_int("TOP_K", 10)
    als_factors: int = _env_int("ALS_FACTORS", 32)
    als_iters: int = _env_int("ALS_ITERS", 15)
    hybrid_weights: tuple[float, float, float] = tuple(  # type: ignore[assignment]
        float(x) for x in os.environ.get("HYBRID_WEIGHTS", "0.4,0.4,0.2").split(",")
    )

    # Arquivos
    @property
    def file_alunos(self) -> Path:
        return self.data_dir / "dados_alunos.csv"

    @property
    def file_materiais(self) -> Path:
        return self.data_dir / "materiais_didaticos.csv"

    @property
    def file_interacoes(self) -> Path:
        return self.data_dir / "interacoes.csv"

    # Artefatos
    @property
    def artifact_path(self) -> Path:
        return self.artifacts_dir / "model_bundle.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"


settings = Settings()
