from __future__ import annotations

import numpy as np

from src.preprocessing import build_index_maps, build_rating_matrix
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.knowledge import KnowledgeRecommender


def test_content_recommender_fit_e_recomenda(alunos_mini, materiais_mini, interacoes_mini):
    idx = build_index_maps(alunos_mini, materiais_mini)
    rec = ContentBasedRecommender(max_features=50).fit(alunos_mini, materiais_mini, interacoes_mini, idx)
    scores = rec.score_user(0)
    assert scores.shape == (idx.n_items,)
    top, sc = rec.recommend(0, k=3)
    assert len(top) == 3
    assert sc[0] >= sc[-1]


def test_collab_recommender_recomenda_e_predict(alunos_mini, materiais_mini, interacoes_mini):
    idx = build_index_maps(alunos_mini, materiais_mini)
    R = build_rating_matrix(interacoes_mini, idx)
    rec = CollaborativeRecommender(n_factors=2, n_iter=5, random_state=0).fit(R, idx)
    scores = rec.score_user(0)
    assert scores.shape == (idx.n_items,)
    pred = rec.predict_rating(0, 0)
    assert 1.0 <= pred <= 5.0


def test_knowledge_recommender_pontuacao_areas(alunos_mini, materiais_mini, interacoes_mini):
    idx = build_index_maps(alunos_mini, materiais_mini)
    rec = KnowledgeRecommender().fit(alunos_mini, materiais_mini, idx)
    # Aluno 1: período 3 -> Iniciante; áreas IA, ML
    scores_a1 = rec.score_user(0)
    # material 10 (IA, Intermediário) deve marcar interesse mas não mesmo nível
    # material 50 (Programação, Iniciante) sem area de interesse mas nível bate
    assert scores_a1.sum() > 0


def test_hybrid_combina_pesos(alunos_mini, materiais_mini, interacoes_mini):
    idx = build_index_maps(alunos_mini, materiais_mini)
    R = build_rating_matrix(interacoes_mini, idx)
    content = ContentBasedRecommender(max_features=50).fit(alunos_mini, materiais_mini, interacoes_mini, idx)
    collab = CollaborativeRecommender(n_factors=2, n_iter=5, random_state=0).fit(R, idx)
    knowledge = KnowledgeRecommender().fit(alunos_mini, materiais_mini, idx)
    hybrid = HybridRecommender(content, collab, knowledge, weights=(0.5, 0.3, 0.2))
    s = hybrid.score_user(1)
    assert s.shape == (idx.n_items,)
    # scores pós-normalização ficam em [0, 1] em cada componente -> soma <= 1
    assert s.max() <= 1.0 + 1e-6
    # pesos somam 1
    assert np.isclose(hybrid.weights.sum(), 1.0)
