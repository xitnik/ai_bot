## Как запускать проверки
- Юнит/интеграционные тесты: `pytest`
- Eval golden-наборов: `python -m tools.eval_runner` (пороги A1/A2)
- Линтер: `ruff check .`

## Примечание
По умолчанию используются fake-интеграции (Telegram/MAX/Avito/Bitrix24/1С). Для реальных стендов заполните `.env`.
