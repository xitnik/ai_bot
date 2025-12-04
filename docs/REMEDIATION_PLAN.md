# REMEDIATION_PLAN

## Формат REM-карточек
- ID: REM-XXX
- Goal: (A1/A2/A3/Acceptance)
- Type: CODE / TEST / DOC / OBSERVABILITY / INFRA
- Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО / ТРЕБУЕТ ВНЕШНИЙ СЕРВИС
- DoD: критерий готовности
- Files: целевые файлы/директории

## Backlog
- REM-001 | Goal: A1 | Type: CODE/DOC | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: единый конфиг (.env.example, loader), перечислены обязательные env vars.  
  Files: `.env.example`, `config.py`, `docs/ARCHITECTURE.md`.

- REM-002 | Goal: A1 Integrations | Type: CODE/TEST | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: интерфейсы Telegram/MAX/Avito/Bitrix24/1С с fake-реализациями и контрактными тестами.  
  Files: `integrations/*`, `tests/test_integrations_fakes.py`.

- REM-003 | Goal: A1 RAG/NER | Type: CODE/TEST | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: офлайн retrieval + ingestion, baseline NER, фикстуры и тесты.  
  Files: `rag/*`, `ner.py`, `fixtures/rag_docs/*`, `tests/test_rag_*.py`, `tests/test_ner.py`.

- REM-004 | Goal: A2 Agents/Supervisor/CB/A/B | Type: CODE/TEST | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: единый контракт Agent, добавлен калькулятор, supervisor с A/B, circuit breaker с fallback, тесты.  
  Files: `agents/*`, `orchestrator.py`, `tests/test_supervisor.py`.

- REM-005 | Goal: A1/A2 LoRA | Type: CODE/DOC/TEST | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: загрузка адаптера по роли с fallback, toy training pipeline e2e с perplexity/квал. метриками.  
  Files: `training/*`, `sales_agent.py`, `docs/TRAINING_PIPELINE.md`, `tests/test_training_pipeline.py`.

- REM-006 | Goal: A1/A2 Observability | Type: OBSERVABILITY/CODE/TEST | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: structured logging, trace_id propagation, /metrics endpoint, метрики latency/success, Grafana JSON.  
  Files: `observability.py`, `metrics.py`, `main.py`, `sales_agent.py`, `dashboards/grafana/dashboard.json`, `tests/test_metrics.py`.

- REM-007 | Goal: Acceptance A1/A2 eval | Type: TEST/CODE | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: golden-набор intents/агентов, eval-раннер с порогами ≥80%/≥95%, возвращает non-zero при провале.  
  Files: `fixtures/golden_*`, `tools/eval_runner.py`, `tests/test_eval_runner.py`.

- REM-008 | Goal: A3 Infra/Docs/Handover | Type: DOC/INFRA | Status: МОГУ СДЕЛАТЬ ЛОКАЛЬНО  
  DoD: docker-compose/OTLP placeholders, handover templates, ops/runbook/architecture docs, CI workflow.  
  Files: `docs/*`, `.github/workflows/ci.yml`, `docs/templates/*`, `docs/HANDOVER_CHECKLIST.md`.
