## Quickstart
```bash
poetry install
poetry run pip install torch --index-url https://download.pytorch.org/whl/cpu
poetry run python -m engine_rl.agents.td3_train
poetry run python -m engine_rl.agents.eval_policies  # графики + метрики

# 6) Куда дальше (последовательно)
1) Тюнинг коэффициентов потерь/материалов (под реальные числа из статьи).
2) Расширение действия до `[throttle, fuel]` и задача: **трекинг оборотов** + удержание AFR.
3) Бейзлайны PID/MPC для сравнения (и NN≈MPC как опция).
4) Экспорт актёра: ONNX → TFLite (int8) + замер latency/памяти (`docs/mcu.md`).
5) Логи экспериментов (W&B/MLflow) — можно подключить флагом.

Если хочешь, я могу сразу сгенерить минимальный `README` и `docs/experiments.md` с шаблоном таблиц метрик, чтобы в репе всё выглядело академично с первого пуша.
