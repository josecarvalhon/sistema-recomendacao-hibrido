# Sistema de Recomendação Híbrido — Plataforma de Ensino Superior

API REST que recomenda materiais didáticos (livros, artigos, vídeos) para alunos de uma plataforma de ensino, combinando filtragem baseada em conteúdo, filtragem colaborativa e regras de conhecimento. Empacotado em Docker e treinado offline a partir de três datasets em CSV.

## Sumário

1. Visão geral da arquitetura
2. Decisões de projeto e tratamento dos dados
3. Modelos de recomendação
4. Métricas de avaliação
5. Como executar (Docker e local)
6. Endpoints da API
7. Resultados obtidos
8. Pontos fortes e fracos
9. Sugestões de melhorias
10. Estrutura do repositório

---

## 1. Visão geral da arquitetura

O sistema é composto por dois processos distintos orquestrados via Docker Compose. O serviço `trainer` é responsável pelo pipeline offline, executado uma única vez (ou periodicamente): carrega os três CSVs, aplica os tratamentos de qualidade de dados, treina os três modelos individuais e o combinador híbrido, calcula as métricas de avaliação sobre um split de teste por usuário e persiste tudo em dois artefatos versionáveis (`artifacts/model_bundle.joblib` e `artifacts/metrics.json`). O serviço `api` é uma aplicação FastAPI executada com Uvicorn que carrega esses artefatos no startup e serve recomendações sob demanda em latência baixa.

A separação treino-serviço segue o padrão clássico de sistemas de recomendação em produção, em que o custo do ajuste do modelo é pago uma vez no batch e a inferência online se reduz a multiplicações de matrizes pequenas. O Compose foi configurado de modo que o `api` só sobe após a conclusão bem-sucedida do `trainer` (`depends_on.condition: service_completed_successfully`), evitando que a API tente carregar artefatos inexistentes.

A camada de modelagem é organizada em torno de uma interface comum (`BaseRecommender`) com um único método `score_user(user_idx) -> np.ndarray`, devolvendo a pontuação de todos os itens para o usuário. Os quatro modelos disponíveis (`content`, `collab`, `knowledge` e `hybrid`) implementam essa interface, o que permite avaliar e servir qualquer um deles sem código duplicado.

## 2. Decisões de projeto e tratamento dos dados

A análise exploratória inicial dos três CSVs revelou um desalinhamento estrutural entre o log de interações e os dois cadastros. Em `interacoes.csv`, o campo `id_aluno` cobre a faixa 1–100.000 e o campo `id_material` cobre 1–50.000, mas os cadastros (`dados_alunos.csv` e `materiais_didaticos.csv`) listam apenas 10.000 entidades cada (IDs 1–10.000). Apenas 2,01% das 10.000.000 interações têm os dois IDs simultaneamente cruzáveis com os cadastros, totalizando 200.622 linhas.

A estratégia adotada foi filtrar o subset cruzável (decisão equivalente a tratar os IDs órfãos como ruído de geração e descartá-los). Essa escolha preserva a coerência conceitual entre as três famílias de modelos — filtragem por conteúdo, colaborativa e baseada em regras —, todas operando sobre o mesmo universo de 10.000 alunos por 10.000 materiais. As 200.622 interações resultantes formam uma matriz com esparsidade de 99,80%, esperada no domínio de recomendação educacional.

O log do treino registra explicitamente o impacto desse filtro, em conformidade com o princípio de auditabilidade. As métricas exportadas em `metrics.json` incluem o bloco `integridade_referencial`, com a quantidade de interações antes e depois do filtro, justamente para que o leitor do relatório possa avaliar o custo dessa decisão. Aplicou-se também um corte mínimo de 5 interações por usuário e por item, removendo itens muito frios que distorceriam a matriz de fatoração; no dataset atual o corte é praticamente inócuo (200.622 → 200.621 linhas).

A divisão treino-teste é feita por usuário, separando 20% das interações de cada aluno com pelo menos duas observações. Esse esquema garante que todo usuário no conjunto de teste tenha sido visto durante o treino, condição necessária para avaliar Precision@K e Recall@K dos quatro modelos sob a mesma população.

## 3. Modelos de recomendação

A filtragem baseada em conteúdo (`ContentBasedRecommender`) representa cada material por um vetor concatenado: TF-IDF com unigramas e bigramas sobre `descricao + titulo + autor` (limitado a 2.000 features) e um one-hot esparso de `tipo`, `area` e `nivel`. O perfil do aluno é construído como combinação igualmente ponderada do perfil cadastral (suas áreas de interesse e o nível esperado a partir do período acadêmico, projetados no mesmo espaço de atributos do material) e do perfil empírico (média ponderada pela avaliação dos vetores dos materiais já consumidos no treino). O score final é o cosseno entre o perfil do aluno e cada material.

A filtragem colaborativa (`CollaborativeRecommender`) é implementada via SVD truncada do scikit-learn aplicada à matriz de avaliações centralizada na média global. A escolha do TruncatedSVD em vez de bibliotecas como `implicit` ou `surprise` foi deliberada: ele é parte da stack padrão de scikit-learn, evita compilação nativa no container e fornece tanto ranking (top-K via produto interno do fator do usuário pelo dos itens) quanto predição de avaliação numérica para o cálculo do RMSE. O número de fatores latentes é configurável (`ALS_FACTORS`, padrão 32) e o algoritmo `randomized` escala bem para a matriz aluno × material atual.

O modelo baseado em conhecimento (`KnowledgeRecommender`) implementa regras simples sobre os atributos do aluno e do material. O score é a soma de três contribuições: aderência da área do material às áreas de interesse declaradas pelo aluno (peso 1,0); aderência da área do material a uma das disciplinas cursadas pelo aluno (peso 0,5, via mapeamento explícito disciplina → área); e compatibilidade entre o nível do material (Iniciante/Intermediário/Avançado) e o nível esperado pelo período acadêmico do aluno, com bônus para coincidência exata, bônus reduzido para nível adjacente e penalidade leve para nível muito distante. O mapeamento período → nível é monotonamente crescente: períodos 2–3 esperam Iniciante, 4–5 Intermediário, 6–8 Avançado.

O combinador híbrido (`HybridRecommender`) normaliza os scores dos três modelos para o intervalo [0, 1] via min-max e aplica uma combinação convexa configurável via variável de ambiente. O padrão (0,4 / 0,4 / 0,2) atribui peso igual a conteúdo e colaborativo e peso menor à camada de regras, refletindo o reconhecimento de que regras simples são mais úteis como desempate e cold-start do que como sinal principal.

## 4. Métricas de avaliação

A avaliação offline cobre o conjunto de métricas listado nos requisitos. **Precision@K, Recall@K e F1@K** são calculados sobre os usuários presentes no conjunto de teste, com K configurável (padrão 10). Para cada usuário, os itens já consumidos no treino são removidos da lista de candidatos, e a pontuação é o quanto dos itens de teste reaparece nas K primeiras recomendações. O cálculo é feito em uma amostra aleatória de até 2.000 usuários para manter o tempo de avaliação na ordem de poucos segundos por modelo.

O **RMSE** é exclusivo do recomendador colaborativo, que produz predição numérica. A predição é truncada ao intervalo [1, 5] antes de comparar com a avaliação observada no conjunto de teste.

A **cobertura do catálogo** (Coverage) é definida como a fração de itens que aparecem em alguma top-K em toda a amostra avaliada. É um indicador clássico de viés de popularidade: um modelo que sempre recomenda os mesmos blockbusters terá Coverage baixo, ainda que com Precision aceitável.

A **diversidade** intra-lista (Diversity) usa a média de (1 − cosseno) par-a-par dentro da top-K de cada usuário, no espaço de atributos textuais e categóricos do material (mesmas features usadas pelo recomendador de conteúdo). Valores próximos de zero indicam recomendações altamente redundantes; valores próximos de um, listas com itens muito distintos.

As métricas online (CTR, tempo de engajamento, satisfação) estão fora do escopo do entregável atual por requererem instrumentação da plataforma; a documentação reserva espaço para elas no `metrics.json`, e a API expõe os endpoints necessários para coletar logs de impressão e clique em uma fase futura.

## 5. Como executar

### Docker (recomendado)

Antes de tudo, copie `.env.example` para `.env` e ajuste `DATA_HOST_DIR` para apontar para a pasta local que contém os três CSVs (`dados_alunos.csv`, `materiais_didaticos.csv`, `interacoes.csv`).

```bash
cp .env.example .env
# editar .env com o caminho correto

docker compose up --build
```

O serviço `trainer` executa o pipeline e termina; em seguida o `api` sobe na porta 8000. Após a primeira execução, é possível subir apenas a API sem retreinar:

```bash
docker compose up api
```

E para rodar somente o treino isoladamente:

```bash
docker compose run --rm trainer
```

### Local (sem Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DATA_DIR=/caminho/para/csvs
export ARTIFACTS_DIR=./artifacts

python -m scripts.train
uvicorn src.api:app --reload --port 8000
```

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `DATA_DIR` | `/app/data` | Diretório com os três CSVs |
| `ARTIFACTS_DIR` | `/app/artifacts` | Onde salvar `model_bundle.joblib` e `metrics.json` |
| `TEST_FRACTION` | `0.2` | Fração de teste por usuário |
| `MIN_INTERACTIONS_USER` | `5` | Corte mínimo de interações por aluno |
| `MIN_INTERACTIONS_ITEM` | `5` | Corte mínimo de interações por material |
| `TOP_K` | `10` | K usado nas métricas e padrão da API |
| `ALS_FACTORS` | `32` | Fatores latentes do SVD |
| `ALS_ITERS` | `15` | Iterações da decomposição |
| `HYBRID_WEIGHTS` | `0.4,0.4,0.2` | Pesos: conteúdo, colaborativo, conhecimento |
| `RANDOM_SEED` | `42` | Semente de aleatoriedade |

## 6. Endpoints da API

A documentação interativa OpenAPI fica em `http://localhost:8000/docs` após subir a API. Os endpoints principais:

```
GET /health
GET /info
GET /metrics
GET /alunos/{id_aluno}
GET /alunos/{id_aluno}/historico?limit=20
GET /materiais/{id_material}
GET /recomendacoes/{id_aluno}?k=10&strategy=hybrid|content|collab|knowledge&excluir_consumidos=true
```

Exemplo de chamada:

```bash
curl 'http://localhost:8000/recomendacoes/1?k=5&strategy=hybrid'
```

Resposta (resumida):

```json
{
  "id_aluno": 1,
  "strategy": "hybrid",
  "k": 5,
  "items": [
    {"id_material": 6052, "titulo": "Aprendendo Python", "tipo": "livro",
     "area": "Programação", "nivel": "Iniciante", "autor": "Eric Matthes",
     "score": 0.911}
  ]
}
```

O parâmetro `strategy` permite comparar diretamente as quatro abordagens para o mesmo aluno, útil para inspeção qualitativa e A/B test offline.

## 7. Resultados obtidos

Os números abaixo correspondem à execução com sementes e hiperparâmetros padrão, sobre uma amostra de 2.000 usuários do conjunto de teste, K=10. Os valores estão em `artifacts/metrics.json` após o treino.

| Modelo | Precision@10 | Recall@10 | F1@10 | Coverage | Diversity | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Content | 0,00015 | 0,00031 | 0,00020 | 0,507 | 0,000 | — |
| Collab  | 0,00050 | 0,00124 | 0,00071 | 0,096 | 0,729 | 1,414 |
| Knowledge | 0,00020 | 0,00039 | 0,00026 | 0,149 | 0,018 | — |
| Hybrid  | 0,00035 | 0,00091 | 0,00051 | 0,230 | 0,122 | — |

Os valores absolutos de Precision e Recall são propositalmente baixos e merecem leitura cuidadosa. Os três datasets fornecidos têm avaliação distribuída de forma aproximadamente uniforme entre 1 e 5 (média 3,00, desvio 1,41), o que é consistente com geração sintética sem padrão real subjacente. O RMSE do colaborativo (1,414) é numericamente igual ao desvio-padrão dessa distribuição, confirmando que o modelo está, de fato, próximo de prever a média global — não há sinal preditivo a ser extraído de avaliações independentes e identicamente distribuídas. As métricas offline aqui têm valor metodológico (validam o pipeline) mas não devem ser interpretadas como qualidade do recomendador em produção; com dados reais de engajamento, a expectativa é que Precision e Recall fiquem ordens de grandeza acima.

A leitura comparativa entre modelos, ainda assim, é informativa. O colaborativo lidera Precision@10 e Recall@10 dentro do que é estatisticamente possível extrair, mas concentra suas recomendações em poucos itens populares (Coverage 9,6%). O recomendador de conteúdo cobre metade do catálogo, mas com Diversity quase zero — efeito direto da estrutura do dataset, em que existem 1.000 cópias por autor, todas com `descricao`, `titulo` e `autor` idênticos, gerando vetores TF-IDF empatados. O modelo de regras tem comportamento intermediário, com forte tendência a empilhar materiais da mesma área e nível. O híbrido equilibra os três sinais e termina com Coverage de 23%, valor superior à média ponderada simples dos três, indicando que a combinação efetivamente diversifica o ranking.

## 8. Pontos fortes e fracos

Entre os pontos fortes, destaca-se a separação clara entre treino e serviço, com artefato único e versionável, o que torna o pipeline reprodutível e fácil de auditar; a interface comum dos quatro recomendadores, que permite trocar a estratégia via parâmetro de query sem código duplicado; o tratamento explícito do desalinhamento referencial, registrado nas métricas exportadas; e a ausência de dependências nativas além de scikit-learn, simplificando o build do Docker.

Entre os pontos fracos, o mais relevante é a limitação imposta pela qualidade do dataset: avaliações sintéticas uniformes não permitem que nenhum modelo aprenda padrão preditivo, e as métricas absolutas refletem isso. Em segundo lugar, a redundância no catálogo (1.000 materiais com mesmo título/autor por bloco) inflaciona artificialmente as listas de top-K do modelo de conteúdo — em produção, seria de-duplicado por título canônico. Em terceiro, a etapa de RMSE só é avaliada para o colaborativo; modelos de conteúdo e regras não produzem rating numérico, então a comparação é parcial. Por fim, o split é aleatório por usuário, e não cronológico — uma versão mais rigorosa usaria a coluna `data` para simular o cenário em que se prevê o futuro a partir do passado.

## 9. Sugestões de melhorias

Em ordem de impacto esperado: (i) adotar split temporal usando `data` como eixo, com janela móvel para simular avaliação contínua; (ii) substituir o SVD truncado por ALS implícito (biblioteca `implicit`) sobre uma matriz de confiança ponderada por `duracao_minutos × tipo_interacao`, que captura melhor sinais implícitos de interesse; (iii) de-duplicar o catálogo por chave canônica (título + autor + tipo) antes de gerar recomendações, evitando listas com cinco cópias do mesmo livro; (iv) calibrar os pesos do híbrido via grid-search no conjunto de validação, em vez do default 0,4/0,4/0,2; (v) adicionar coleta de eventos de impressão e clique na API para viabilizar métricas online (CTR, tempo de engajamento), o que requer expor um endpoint POST `/eventos` e persistir em base apropriada; (vi) introduzir Postgres + Redis no Compose para servir o histórico do aluno em tempo real e cachear top-K dos usuários mais ativos; (vii) treinar embeddings densos da `descricao` com sentence-transformers em substituição ao TF-IDF, melhorando a similaridade semântica entre materiais.

## 10. Estrutura do repositório

```
.
├── Dockerfile                     # imagem única usada por trainer e api
├── docker-compose.yml             # orquestração trainer -> api
├── .env.example                   # template de variáveis de ambiente
├── .dockerignore
├── requirements.txt
├── README.md                      # este documento
├── data/                          # CSVs (montados via volume)
├── artifacts/                     # gerados pelo trainer
│   ├── model_bundle.joblib
│   └── metrics.json
├── scripts/
│   └── train.py                   # entry-point do pipeline offline
├── src/
│   ├── config.py                  # leitura de env vars
│   ├── data_loader.py             # carregamento e filtro de integridade
│   ├── preprocessing.py           # index maps, splits, matrizes esparsas
│   ├── evaluation.py              # Precision@K, Recall@K, F1, RMSE, Cov, Div
│   ├── api.py                     # FastAPI
│   └── recommenders/
│       ├── base.py                # interface BaseRecommender
│       ├── content_based.py
│       ├── collaborative.py
│       ├── knowledge.py
│       └── hybrid.py
└── tests/
    ├── conftest.py
    ├── test_data_loader.py
    ├── test_recommenders.py
    └── test_api.py
```
