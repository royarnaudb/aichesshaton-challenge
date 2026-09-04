"""Run clocked matches and report strength plus search performance metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import INCREMENT_MS, PLY_CAP
from harness.sandbox import RUNNER, Agent

METRIC_LINE = re.compile(r"METRIC (\{.*\})$")


class MeasuredAgent(Agent):
    """The normal sandbox agent with test-only timing and metric collection."""

    def __init__(self, directory: Path, profile: str | None = None) -> None:
        super().__init__([sys.executable, str(RUNNER), str(directory.resolve())])
        self.profile = profile
        self.init_ms = 0.0
        self.move_ms: list[float] = []

    def start(self, init_budget_s: float) -> None:
        started = time.perf_counter()
        previous_profile = os.environ.get("CHESS_TIME_PROFILE")
        if self.profile:
            os.environ["CHESS_TIME_PROFILE"] = self.profile
        try:
            super().start(init_budget_s)
        finally:
            if previous_profile is None:
                os.environ.pop("CHESS_TIME_PROFILE", None)
            else:
                os.environ["CHESS_TIME_PROFILE"] = previous_profile
        self.init_ms = (time.perf_counter() - started) * 1000.0

    def move(self, fen: str, time_left_ms: int) -> str:
        started = time.perf_counter()
        try:
            return super().move(fen, time_left_ms)
        finally:
            self.move_ms.append((time.perf_counter() - started) * 1000.0)

    def search_metrics(self) -> list[dict[str, int]]:
        records: list[dict[str, int]] = []
        for line in self.stderr_tail.splitlines():
            match = METRIC_LINE.fullmatch(line.strip())
            if match:
                try:
                    record = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and all(isinstance(v, int) for v in record.values()):
                    records.append(record)
        return records


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentage * len(ordered)) - 1)
    return ordered[index]


def elo_estimate(score: float, opponent_rating: float = 1500.0) -> float:
    bounded = min(0.99, max(0.01, score))
    return opponent_rating + 400.0 * math.log10(bounded / (1.0 - bounded))


def summarize(agent: Path, opponent: Path, games: int, wins: int, draws: int, losses: int,
              terminations: Counter[str], measured: list[MeasuredAgent]) -> dict[str, Any]:
    score = (wins + draws / 2.0) / games
    timings = [duration for item in measured for duration in item.move_ms]
    records = [record for item in measured for record in item.search_metrics()]
    nodes = [record["nodes"] for record in records if "nodes" in record]
    depths = [record["depth"] for record in records if "depth" in record]
    result: dict[str, Any] = {
        "agent": str(agent),
        "opponent": str(opponent),
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_estimate_vs_1500": elo_estimate(score),
        "terminations": dict(terminations),
        "failed_games": sum(
            count for name, count in terminations.items() if name in FAILED_TERMINATIONS
        ),
        "moves": len(timings),
        "latency_ms": {
            "mean": sum(timings) / len(timings) if timings else 0.0,
            "p50": percentile(timings, 0.50),
            "p95": percentile(timings, 0.95),
            "max": max(timings, default=0.0),
        },
        "search": {
            "metric_records": len(records),
            "mean_nodes": sum(nodes) / len(nodes) if nodes else 0.0,
            "max_depth": max(depths, default=0),
            "mean_depth": sum(depths) / len(depths) if depths else 0.0,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/minimax"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--profile",
        choices=("balanced", "conservative", "tactical", "endgame"),
        default="balanced",
    )
    arguments = parser.parse_args()
    os.environ["CHESS_TIME_PROFILE"] = arguments.profile
    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    wins = draws = losses = 0
    terminations: Counter[str] = Counter()
    measured: list[MeasuredAgent] = []

    for game in range(arguments.games):
        plays_white = game % 2 == 0
        white_dir, black_dir = (agent, opponent) if plays_white else (opponent, agent)
        white = MeasuredAgent(white_dir)
        black = MeasuredAgent(black_dir)
        outcome = play_match(
            white, black, arguments.base_ms, arguments.increment_ms, arguments.ply_cap
        )
        measured.append(white if plays_white else black)
        terminations[outcome.termination] += 1
        if outcome.result == "draw" or outcome.result == "void":
            draws += 1
        elif (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1
        print(f"game {game + 1}/{arguments.games}: {outcome.result} by {outcome.termination}")

    result = summarize(
        agent, opponent, arguments.games, wins, draws, losses, terminations, measured
    )
    print(json.dumps(result, indent=2))
    if arguments.json_out:
        arguments.json_out.write_text(json.dumps(result, indent=2) + "\n")
    if result["failed_games"]:
        raise SystemExit("agent failed one or more games")


if __name__ == "__main__":
    main()
