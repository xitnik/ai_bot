# HANDOVER_CHECKLIST

- [ ] Кодовая база и тесты в репозитории Заказчика.
- [ ] Обновленные конфиги (.env.example, litellm_config.yaml) без секретов.
- [ ] Артефакты моделей/LoRA: `artifacts/lora_adapter.json`, `training_metrics.json`.
- [ ] Документация: ARCHITECTURE, RUNBOOK, OPERATIONS, TRAINING_PIPELINE, templates/ACT + REPORT.
- [ ] Дашборды: `dashboards/grafana/dashboard.json` (провиженинг).
- [ ] Eval отчеты (A1/A2): вывод `tools/eval_runner.py`.
- [ ] Настроенные интеграции: токены Telegram/MAX/Avito, Bitrix24 webhook, 1С URL; подтверждение тестов.
- [ ] CI рабочий (линтеры, тесты, eval).
