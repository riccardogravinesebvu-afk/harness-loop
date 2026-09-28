default:
    @just --list

sync:
    uv sync --group dev

lint:
    uv run ruff check . --fix && uv run ruff format .

test:
    uv run pytest -q

evals seed="42":
    uv run python -m evals.run --seed {{seed}}

select:
    uv run python -m evals.dataset select

dataset-report:
    uv run python -m evals.dataset report

generate *flags:
    uv run python -m evals.generate {{flags}}

loop max="12" base="main" flags="":
    uv run python -m src.optimizer.loop --max-iterations {{max}} --base {{base}} {{flags}}

feedback port="8765":
    uv run uvicorn src.feedback.api:app --port {{port}}

replay-refused:
    uv run python -m scripts.replay_refused

