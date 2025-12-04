# AUDIT_REPORT

## Обзор стека и артефактов
- FastAPI сервисы: `main.py` (webchat gateway), `sales_agent.py` (агент-продавец), `alternatives_agent.py` (агент альтернатив), `procurement/api.py` (закупки).
- Инфраструктура/конфиги: отсутствуют docker/helm; только `pyproject.toml`, `litellm_config.yaml`.
- БД: SQLite/Postgres через `db.py`, `procurement/db.py`.
- Observability: `events_logger.py` (в БД), `otel.py` (in-memory OTEL), структурированные логи отсутствуют.
- Тесты: `tests/` (e2e webchat, sales агент, alternatives, procurement, vector index).

## Трассировка требований A1/A2/A3
- Интеграции Telegram/MAX/Авито/Bitrix24/1С: **НЕ НАЙДЕНО**. Есть только обращение к `/1c_api/stock` в `sales_tools_clients.py` без клиента.
- RAG/NER: **ЧАСТИЧНО**. In-memory vector store (`vector_index.py`), embeddings через LiteLLM/OpenAI (`embeddings_client.py`); нет ingestion/ретривера/NER.
- Агент-продавец с LoRA: **ЧАСТИЧНО**. `sales_agent.py` использует модель `yagpt-lora-sales` в конфиге `litellm_config.yaml`, но нет адаптера/пайплайна.
- Метрики/дашборды: **НЕ НАЙДЕНО**. Только запись событий в таблицу events.
- Мультиагентность A2 (калькулятор, альтернативы, закупщик): альтернативы и закупки есть; калькулятора нет, supervisor/circuit breaker отсутствуют; A/B стратегии нет.
- Observability A2: централизованные логи/tracing отсутствуют (есть in-memory OTEL).
- A3 пайплайн LoRA: **НЕ НАЙДЕНО**. Нет обучения, дрейф-контроля, perplexity/offline метрик, Grafana/Prometheus/IaC.
- Документация/CI: README отсутствует, нет CI/CD.

## Риски и замечания
- Нет внешних интеграций и их тестов; отсутствует идемпотентность CRM/1С.
- Нет единой конфигурации (.env, схема).
- Нет structured logging с trace_id/request_id.
- Нет golden-наборов intents/агентов и автооценки acceptance порогов.
- Отсутствуют дашборды/экспортеры метрик и трассировок.
