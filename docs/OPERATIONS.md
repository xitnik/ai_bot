# OPERATIONS

## Ежедневные задачи
- Проверять eval отчеты (`python -m tools.eval_runner`) — целевые пороги: overall >=0.8, per-intent >=0.95.
- Просмотр `/metrics` и событий `events` для деградаций агентов/интеграций.
- Обновление golden наборов в `fixtures/` при изменении интентов/агентов.

## Обновление конфигурации
- Все обязательные env перечислены в `.env.example`.
- Для переключения интеграций на реальные стенды задать токены и URL (Telegram/MAX/Avito/Bitrix24/1С).
- Для OTEL/Prometheus указать `OTEL_EXPORTER_OTLP_ENDPOINT` и подключить collector.

## Релизы
- CI (см. `.github/workflows/ci.yml`) запускает линтеры/тесты/eval.
- Рекомендуемый артефакт: контейнеры сервисов + `artifacts/lora_adapter.json` (если переобучалась LoRA).

## Документы для передачи
- Архитектура: `docs/ARCHITECTURE.md`
- Runbook/Operations: `docs/RUNBOOK.md`, `docs/OPERATIONS.md`
- Training: `docs/TRAINING_PIPELINE.md`
- Handover: `docs/HANDOVER_CHECKLIST.md`, `docs/templates/*`
