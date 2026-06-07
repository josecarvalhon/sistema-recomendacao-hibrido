"""Smoke test da API sem artefato treinado: /health responde degraded."""
from __future__ import annotations

import os

# Aponta artefatos para um diretório temporário antes de importar a app
os.environ["ARTIFACTS_DIR"] = "/tmp/reco_test_artifacts_inexistente"
os.environ["DATA_DIR"] = "/tmp/reco_test_data_inexistente"

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_sem_artefato():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")


def test_recomendacao_sem_artefato_retorna_503():
    r = client.get("/recomendacoes/1?k=5")
    assert r.status_code == 503
