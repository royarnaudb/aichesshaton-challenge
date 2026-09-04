SHELL := /bin/bash

.PHONY: setup play arena metrics elo tournament test zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

metrics:
	uv run python -m tools.metrics --opponent baselines/minimax --games 20 --json-out metrics.json

tournament:
	uv run python -m tools.tournament . baselines/random baselines/greedy baselines/minimax --games-per-pair 4 --json-out tournament.json

elo:
	uv run python -m tools.tournament . baselines/random baselines/greedy baselines/minimax baselines/numba --games-per-pair 20 --base-ms 10000 --increment-ms 100 $(if $(STOCKFISH),--stockfish "$(STOCKFISH)") --json-out elo.json

test:
	uv run python -m unittest discover -s tests -v

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
