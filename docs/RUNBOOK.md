# RUNBOOK

## Локальный запуск
```
uvicorn main:app --reload
uvicorn sales_agent:app --reload --port 8000
uvicorn alternatives_agent:app --reload --port 8001
uvicorn procurement.api:router --reload --port 8002  # через FastAPI include
```

## Где смотреть логи и трассировки
- STDOUT с JSON (`observability.py`), ключи: `service`, `trace_id`, `session_id`, `user_id`.
- OTEL: in-memory (`otel.py`), экспорт OTLP по env `OTEL_EXPORTER_OTLP_ENDPOINT` при подключении коллектора.
- События пайплайна: таблица `events` (`db.py`).

## Метрики
- Endpoint `/metrics` (Prometheus формат, базовые счетчики/latency averages в `metrics.py`).
- Метрики агентов: `sales_agent_calls_total`, `requests_total` в gateway.

## Инциденты и действия
- Circuit breaker срабатывает → ответы `status=open_circuit`; проверить внешние агенты/интеграции, смотреть `agents/circuit_breaker.py`.
- Eval падает (`tools/eval_runner.py`) → сверить golden-набор `fixtures/golden_intents.json`, корректировать правила в `agents/supervisor.py`.
- 1С/CRM проблемы: включить fake-режим через env (использовать токены вида `fake-*` или localhost) и проверить `integrations/fake.py`.

## БД
- По умолчанию SQLite (`DATABASE_URL`, `PROCUREMENT_DATABASE_URL`), миграций нет — таблицы создаются на старте.
