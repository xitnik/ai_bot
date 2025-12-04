# Bot Suite (LLM Agents)

## Запуск локально
1. Скопируйте `.env.example` в `.env` и задайте нужные переменные (по умолчанию — fake/mock режимы).
2. Установите зависимости:
   ```
   pip install -e .
   ```
3. Запустите сервисы:
   ```
   uvicorn main:app --reload --port 9000
   uvicorn sales_agent:app --reload --port 8000
   uvicorn alternatives_agent:app --reload --port 8001
   uvicorn procurement.api:router --reload --port 8002
   ```
4. Метрики доступны по `/metrics` (gateway), здоровье LLM — `/llm/health` (app.py).

## Тесты и eval
```
pytest
python -m tools.eval_runner
```

## Структура
- `main.py` — gateway, сессии/логирование/метрики.
- `sales_agent.py` — агент-продавец, LoRA-стилизация.
- `alternatives_agent.py`, `procurement/*` — доп. агенты.
- `integrations/*` — интерфейсы и моки каналов/CRM/1С.
- `rag/*`, `ner.py` — baseline RAG/NER.
- `training/*` — toy LoRA pipeline.
- `docs/*` — архитектура, операционные инструкции, шаблоны актов/отчетов.

## Acceptance auto-checks
- Golden intents: `fixtures/golden_intents.json`, пороги >=0.8/0.95 через `tools/eval_runner.py`.
- Circuit breaker и A/B в `agents/supervisor.py` + `agents/circuit_breaker.py`.
- Grafana шаблон: `dashboards/grafana/dashboard.json`.
