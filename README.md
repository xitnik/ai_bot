# Bot Suite (LLM Agents)

## Запуск локально
1. Скопируйте `.env.example` в `.env` и задайте нужные переменные (по умолчанию — fake/mock режимы).
2. Установите зависимости:
   ```
   pip install -e .
   # для ingest/RAG дополнительных форматов добавьте extras:
   pip install -e ".[ingest,retrieval]"
   ```
3. Запустите сервисы:
   ```
   uvicorn main:app --reload --port 9000
   uvicorn sales_agent:app --reload --port 8000
   uvicorn alternatives_agent:app --reload --port 8001
   uvicorn procurement.api:router --reload --port 8002
   uvicorn agents.pricing_agent:app --reload --port 8003
   ```
4. Метрики доступны по `/metrics` (gateway), здоровье LLM — `/llm/health` (app.py).

## Тесты и eval
```
pytest
python -m tools.eval_runner
```

## RAG
- Чанкинг token-aware в `rag/chunking.py`, ingest через `rag/ingest.py` (extras `.[ingest]`).
- Векторное хранилище выбирается через `RAG_VECTOR_BACKEND` (`chroma`/`memory`/`mysql`) и `RAG_VECTOR_PERSIST_PATH`; extras `.[retrieval]` подтягивает chromadb/sentence-transformers.
- Эмбеддинги: `RAG_EMBEDDING_MODEL` (ai-forever/FRIDA, BAAI/bge-m3, intfloat/multilingual-e5-large, Qwen3-Embedding-8B); sanity-check размера в клиенте.
- Реранкер и генерация: `RAG_RERANKER_ENABLED`, `RAG_RERANKER_MODEL`, `RAG_LLM_MODEL`, `RAG_LLM_TEMPERATURE`, `RAG_LLM_MAX_TOKENS`, топы `RAG_RETRIEVER_KNN_TOP_K`/`RAG_RETRIEVER_FINAL_TOP_K`.
- API: `POST /rag/query` (см. `rag/api.py`), агентский hook `rag/tool.py::RagTool`.

## Структура
- `main.py` — gateway, сессии/логирование/метрики.
- `sales_agent.py` — агент-продавец, LoRA-стилизация.
- `agents/pricing_agent.py` — доменный расчет цены (порода/обработка/отходы/скидки).
- `alternatives_agent.py`, `procurement/*` — доп. агенты.
- `integrations/*` — интерфейсы и моки каналов/CRM/1С.
- `rag/*`, `ner.py` — baseline RAG/NER.
- `training/*` — toy LoRA pipeline.
- `docs/*` — архитектура, операционные инструкции, шаблоны актов/отчетов.

## Конфигурация БД
- По умолчанию сервисы используют SQLite (`DATABASE_URL`, `PROCUREMENT_DATABASE_URL` из `.env.example`), что позволяет запуск без MySQL-драйверов.
- Для продакшн MySQL задайте `DATABASE_URL`/`PROCUREMENT_DATABASE_URL` или поля `MYSQL_*`/`PROCUREMENT_MYSQL_DB` и установите драйвер `asyncmy`.

## Агент deep research для альтернатив
- Запуск: `uvicorn agents.alternatives_deep_research_agent:app --reload --port 8004`.
- Эндпоинты: `/agents/alternatives/deep_research` (и алиас `/agents/alternatives_deep_research/run`).
- Запрос: JSON вида
  ```json
  {
    "query_text": "oak plank 20x100",
    "base_item_name": "oak plank",
    "domain": "material",
    "use_case": "flooring",
    "constraints": ["in_stock", "budget under $500"],
    "nice_to_have": ["eco certification"],
    "hard_filters": {"species": "oak", "grade": "A", "dimensions": {"length": 100, "width": 20, "thickness": 5}}
  }
  ```
- Ответ: конверт `{"status": "ok", "data": {"summary": "...", "research": <AlternativesDeepResearchResult>}}` со структурой task_understanding → research_plan → alternatives → comparison_summary → final_recommendations.
- Переменные окружения: `DEEP_RESEARCH_MODEL`, `DEEP_RESEARCH_SEARCH_BASE_URL`, `DEEP_RESEARCH_SEARCH_API_KEY`, `DEEP_RESEARCH_SEARCH_TIMEOUT`, `DEEP_RESEARCH_SEARCH_TOP_K`, `DEEP_RESEARCH_SEARCH_MAX_RESULTS`, `DEEP_RESEARCH_TEMPERATURE`, `DEEP_RESEARCH_MAX_OUTPUT_TOKENS`.

## Acceptance auto-checks
- Golden intents: `fixtures/golden_intents.json`, пороги >=0.8/0.95 через `tools/eval_runner.py`.
- Circuit breaker и A/B в `agents/supervisor.py` + `agents/circuit_breaker.py`.
- Grafana шаблон: `dashboards/grafana/dashboard.json`.
