"""Estimate relative Elo ratings for every agent in a round-robin tournament."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import INCREMENT_MS, PLY_CAP
from tools.metrics import MeasuredAgent
from tools.stockfish import StockfishAgent


def expected(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def update_ratings(
    ratings: dict[str, float], first: str, second: str, result: float, k: float
) -> None:
    first_expected = expected(ratings[first], ratings[second])
    change = k * (result - first_expected)
    ratings[first] += change
    ratings[second] -= change


def display_name(path: Path | None, name: str) -> str:
    return "Stockfish" if path is None else Path(name).name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agents", nargs="+", type=Path)
    parser.add_argument("--games-per-pair", type=int, default=4)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--initial-elo", type=float, default=1500.0)
    parser.add_argument("--k-factor", type=float, default=32.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--stockfish", type=Path, help="optional local Stockfish executable")
    arguments = parser.parse_args()
    agents: dict[str, Path | None] = {
        str(path.resolve()): path.resolve() for path in arguments.agents
    }
    if arguments.stockfish:
        stockfish = arguments.stockfish.resolve()
        agents[f"stockfish:{stockfish}"] = None
    if len(agents) < 2:
        raise SystemExit("provide at least two distinct agent directories")

    ratings = {name: arguments.initial_elo for name in agents}
    records = {
        name: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "failed_games": 0}
        for name in agents
    }
    terminations: Counter[str] = Counter()
    names = list(agents)

    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            for game in range(arguments.games_per_pair):
                first_white = game % 2 == 0
                white_name, black_name = (first, second) if first_white else (second, first)
                white = (
                    StockfishAgent(arguments.stockfish)
                    if agents[white_name] is None
                    else MeasuredAgent(agents[white_name])
                )
                black = (
                    StockfishAgent(arguments.stockfish)
                    if agents[black_name] is None
                    else MeasuredAgent(agents[black_name])
                )
                outcome = play_match(
                    white,
                    black,
                    arguments.base_ms,
                    arguments.increment_ms,
                    arguments.ply_cap,
                )
                terminations[outcome.termination] += 1
                records[first]["games"] += 1
                records[second]["games"] += 1

                if outcome.result == "void":
                    records[first]["failed_games"] += 1
                    records[second]["failed_games"] += 1
                    print(
                        f"{display_name(agents[first], first)} vs "
                        f"{display_name(agents[second], second)}: void"
                    )
                    continue
                if outcome.result == "draw":
                    first_score = 0.5
                    records[first]["draws"] += 1
                    records[second]["draws"] += 1
                else:
                    first_won = (outcome.result == "white") == first_white
                    first_score = 1.0 if first_won else 0.0
                    if outcome.termination in FAILED_TERMINATIONS:
                        loser = black_name if outcome.result == "white" else white_name
                        records[loser]["failed_games"] += 1
                    records[first]["wins" if first_won else "losses"] += 1
                    records[second]["losses" if first_won else "wins"] += 1
                update_ratings(ratings, first, second, first_score, arguments.k_factor)
                print(
                    f"{display_name(agents[first], first)} vs "
                    f"{display_name(agents[second], second)}: "
                    f"{outcome.result} by {outcome.termination}"
                )

    for stats in records.values():
        stats["score"] = (stats["wins"] + stats["draws"] / 2) / max(1, stats["games"])

    leaderboard = sorted(
        (
            {"agent": name, "elo": round(ratings[name], 1), **stats}
            for name, stats in records.items()
        ),
        key=lambda item: item["elo"],
        reverse=True,
    )
    result = {
        "games_per_pair": arguments.games_per_pair,
        "base_ms": arguments.base_ms,
        "increment_ms": arguments.increment_ms,
        "k_factor": arguments.k_factor,
        "leaderboard": leaderboard,
        "terminations": dict(terminations),
    }
    print(json.dumps(result, indent=2))
    if arguments.json_out:
        arguments.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
