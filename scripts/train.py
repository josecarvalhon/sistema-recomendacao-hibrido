"""Pipeline de treino completo: carrega dados, treina modelos, avalia e persiste artefatos."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np

# permite execução com `python -m scripts.train`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.data_loader import filter_min_counts, load_raw  # noqa: E402
from src.evaluation import (  # noqa: E402
    _build_test_dict,
    _build_train_dict,
    evaluate_rmse,
    evaluate_topk,
)
from src.preprocessing import (  # noqa: E402
    build_index_maps,
    build_rating_matrix,
    split_train_test,
)
from src.recommenders.collaborative import CollaborativeRecommender  # noqa: E402
from src.recommenders.content_based import ContentBasedRecommender  # noqa: E402
from src.recommenders.hybrid import HybridRecommender  # noqa: E402
from src.recommenders.knowledge import KnowledgeRecommender  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def main() -> None:
    t0 = time.time()
    logger.info("Settings: %s", {
        "data_dir": str(settings.data_dir),
        "artifacts_dir": str(settings.artifacts_dir),
        "test_fraction": settings.test_fraction,
        "min_user": settings.min_interactions_user,
        "min_item": settings.min_interactions_item,
        "top_k": settings.top_k,
        "als_factors": settings.als_factors,
        "als_iters": settings.als_iters,
        "hybrid_weights": settings.hybrid_weights,
    })

    raw = load_raw(settings.data_dir)
    logger.info("Integridade: %s", raw.integridade)

    interacoes = filter_min_counts(
        raw.interacoes,
        min_user=settings.min_interactions_user,
        min_item=settings.min_interactions_item,
    )
    logger.info(
        "Após corte mínimo (>=%d/usuário, >=%d/item): %d interações | %d alunos | %d materiais",
        settings.min_interactions_user,
        settings.min_interactions_item,
        len(interacoes),
        interacoes["id_aluno"].nunique(),
        interacoes["id_material"].nunique(),
    )

    # Mantém apenas alunos e materiais que sobraram (para indexação coerente)
    alunos = raw.alunos[raw.alunos["id_aluno"].isin(interacoes["id_aluno"].unique())].reset_index(drop=True)
    materiais = raw.materiais[raw.materiais["id_material"].isin(interacoes["id_material"].unique())].reset_index(drop=True)

    idx = build_index_maps(alunos, materiais)
    logger.info("Index maps: %d alunos x %d materiais", idx.n_users, idx.n_items)

    train_df, test_df = split_train_test(
        interacoes, test_fraction=settings.test_fraction, random_state=settings.random_seed
    )
    rating_train = build_rating_matrix(train_df, idx)

    # ---- Treino dos modelos
    logger.info("Treinando ContentBasedRecommender...")
    content = ContentBasedRecommender().fit(alunos, materiais, train_df, idx)

    logger.info("Treinando CollaborativeRecommender...")
    collab = CollaborativeRecommender(
        n_factors=settings.als_factors,
        n_iter=settings.als_iters,
        random_state=settings.random_seed,
    ).fit(rating_train, idx)

    logger.info("Treinando KnowledgeRecommender...")
    knowledge = KnowledgeRecommender().fit(alunos, materiais, idx)

    hybrid = HybridRecommender(content, collab, knowledge, weights=tuple(settings.hybrid_weights)).fit()

    # ---- Avaliação
    train_dict = _build_train_dict(train_df, idx)
    test_dict = _build_test_dict(test_df, idx)

    rng = np.random.default_rng(settings.random_seed)
    sample = min(2000, len(test_dict))
    logger.info("Avaliando top-K em %d usuários (amostra)", sample)

    item_features = content.item_matrix

    metrics_all = []
    for model in (content, collab, knowledge, hybrid):
        logger.info("  -> %s", model.name)
        m = evaluate_topk(
            model,
            train_dict=train_dict,
            test_dict=test_dict,
            n_items=idx.n_items,
            k=settings.top_k,
            item_features=item_features,
            sample_users=sample,
            rng=rng,
        )
        rmse = evaluate_rmse(collab, test_df, idx) if model.name == "collab" else None
        m_dict = {
            "model": model.name,
            **m,
            "rmse": rmse,
            "k": settings.top_k,
        }
        metrics_all.append(m_dict)
        logger.info("    %s", m_dict)

    # ---- Persistência
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Para o endpoint /historico: tabela compacta com aluno, material, média e contagem
    historico_df = (
        train_df.groupby(["id_aluno", "id_material"])
        .agg(avaliacao_media=("avaliacao", "mean"), n_interacoes=("avaliacao", "size"))
        .reset_index()
    )

    bundle = {
        "idx": idx,
        "alunos": alunos,
        "materiais": materiais,
        "train_dict": train_dict,
        "historico_df": historico_df,
        "content": content,
        "collab": collab,
        "knowledge": knowledge,
        "hybrid_weights": list(settings.hybrid_weights),
        "top_k": settings.top_k,
    }
    joblib.dump(bundle, settings.artifact_path, compress=3)
    logger.info("Artefato salvo em %s (%.1f MB)",
                settings.artifact_path,
                settings.artifact_path.stat().st_size / 1e6)

    metrics_payload = {
        "config": {
            "test_fraction": settings.test_fraction,
            "min_interactions_user": settings.min_interactions_user,
            "min_interactions_item": settings.min_interactions_item,
            "top_k": settings.top_k,
            "als_factors": settings.als_factors,
            "als_iters": settings.als_iters,
            "hybrid_weights": list(settings.hybrid_weights),
            "random_seed": settings.random_seed,
        },
        "integridade_referencial": raw.integridade,
        "subset": {
            "n_users": idx.n_users,
            "n_items": idx.n_items,
            "n_interactions_train": int(len(train_df)),
            "n_interactions_test": int(len(test_df)),
        },
        "models": metrics_all,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    settings.metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False))
    logger.info("Métricas salvas em %s", settings.metrics_path)
    logger.info("Treino concluído em %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
