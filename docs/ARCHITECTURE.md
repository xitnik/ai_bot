# ARCHITECTURE

## Сервисы и entrypoints
- `main.py` — Conversation Gateway `/channels/webchat/message`, метрики `/metrics`.
- `sales_agent.py` — агент-продавец `/agents/sales/run`, использует Planner + Tools + Styler (LoRA-модель `yagpt-lora-<role>`).
- `agents/pricing_agent.py` — доменный расчет цены `/agents/pricing/run` (board feet, породы, обработка, отходы, скидки).
- `alternatives_agent.py` — подбор альтернатив `/agents/alternatives/run`, in-memory vector store.
- `procurement/api.py` — закупки: RFQ создание/парсинг/выбор лучшего оффера.

## Интеграции (ports/adapters)
- `integrations/base.py` — контракты каналов/CRM/1С/маркетплейсов.
- `integrations/clients.py` — Telegram/MAX/Avito/Bitrix24/1С HTTP клиенты с fake fallback.
- `integrations/fake.py` — in-memory моки для локальных тестов и идемпотентности CRM.

## Агенты и оркестрация
- Контракт `agents/base.py`, калькулятор `agents/calculator_agent.py`.
- Supervisor + A/B `agents/supervisor.py`, Circuit Breaker `agents/circuit_breaker.py`.
- Оркестрация `orchestrator.py`: enrich (NER+RAG), route (Supervisor), параллельные вызовы агентов, логирование/трейсы.

## RAG и NER
- RAG ingestion `rag/ingest.py`, retriever `rag/retriever.py`, фикстуры `fixtures/rag_docs`.
- NER baseline `ner.py` (phone/email/ИНН/SKU/числа).

## Observability
- Structured logging `observability.py`, метрики `metrics.py` + `/metrics` endpoint.
- OTEL soft-wrapper `otel.py`, события в БД `events_logger.py`.
- Grafana артефакт: `dashboards/grafana/dashboard.json` (локальный шаблон).

## Модели/LoRA
- Конфиг `litellm_config.yaml`, роль-ориентированная строка модели через `_styler_model_for_role`.
- Toy training pipeline `training/pipeline.py` + сэмплы `training/sample_data.py` → артефакты в `artifacts/`.

## Тесты и eval
- Юниты/интеграции в `tests/` (agents, integrations, RAG/NER, pipeline, eval runner).
- Golden intents `fixtures/golden_intents.json`, eval runner `tools/eval_runner.py` с порогами A1/A2.
