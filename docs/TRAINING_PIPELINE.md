# TRAINING_PIPELINE

## Цель
Офлайн пайплайн (A3) для регулярных дообучений LoRA с минимальными требованиями к окружению.

## Шаги
1. Подготовка данных: диалоги в формате `(user, agent)` — см. `training/sample_data.py`.
2. Запуск обучения:
   ```
   python -m training.pipeline
   ```
   или `run_toy_training(SAMPLE_DIALOGUES, output_dir="artifacts")`.
3. Метрики:
   - Perplexity до/после (`training_metrics.json`: `perplexity_before`, `perplexity_after`).
   - Offline conversion (простая доля совпадений ответов).
4. Артефакт адаптера: `artifacts/lora_adapter.json` (структура и мета).
5. Drift/regularization: сравнить `perplexity_after` с предыдущим запуском, не принимать, если метрика растет.

## Подключение адаптера
- Клиент LLM: модель `yagpt-lora-<role>`; ID адаптера задается env `LORA_ADAPTER_ID`.
- При отсутствии адаптера используется fallback модель без LoRA (см. `_styler_model_for_role` в `sales_agent.py`).

## Расширение до прод-контуров
- Подключить реальный тренировочный кластер и S3/minio для артефактов.
- Добавить OTEL/Prometheus метрики для этапов обучения.
- Планировщик (Cron/Airflow/Prefect) для регулярных задач.
