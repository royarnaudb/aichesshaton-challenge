"""Compare two time-management profiles using the same agent code."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import INCREMENT_MS, PLY_CAP
from tools.metrics import MeasuredAgent
from tools.tournament import update_ratings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument(
        "--first",
        choices=("balanced", "conservative", "tactical", "endgame"),
        default="balanced",
    )
    parser.add_argument(
        "--second",
        choices=("balanced", "conservative", "tactical", "endgame"),
        default="conservative",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--json-out", type=Path, default=Path("profile-match.json"))
    arguments = parser.parse_args()
    agent_path = arguments.agent.resolve()
    ratings = {arguments.first: 1500.0, arguments.second: 1500.0}
    stats = {
        profile: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "failed_games": 0}
        for profile in {arguments.first, arguments.second}
    }
    terminations: Counter[str] = Counter()

    for number in range(arguments.games):
        first_white = number % 2 == 0
        white_profile, black_profile = (
            (arguments.first, arguments.second)
            if first_white
            else (arguments.second, arguments.first)
        )
        white = MeasuredAgent(agent_path, white_profile)
        black = MeasuredAgent(agent_path, black_profile)
        outcome = play_match(
            white, black, arguments.base_ms, arguments.increment_ms, arguments.ply_cap
        )
        terminations[outcome.termination] += 1
        stats[arguments.first]["games"] += 1
        stats[arguments.second]["games"] += 1
        if outcome.result == "void":
            stats[white_profile]["failed_games"] += 1
            stats[black_profile]["failed_games"] += 1
            continue
        if outcome.result == "draw":
            stats[white_profile]["draws"] += 1
            stats[black_profile]["draws"] += 1
            first_score = 0.5
        else:
            winner = white_profile if outcome.result == "white" else black_profile
            loser = black_profile if outcome.result == "white" else white_profile
            stats[winner]["wins"] += 1
            stats[loser]["losses"] += 1
            if outcome.termination in FAILED_TERMINATIONS:
                stats[loser]["failed_games"] += 1
            first_score = 1.0 if winner == arguments.first else 0.0
        update_ratings(ratings, arguments.first, arguments.second, first_score, 32.0)
        if (number + 1) % 10 == 0:
            print(f"completed {number + 1}/{arguments.games}")

    for profile in stats:
        stats[profile]["score"] = (
            stats[profile]["wins"] + stats[profile]["draws"] / 2
        ) / max(1, stats[profile]["games"])

    result = {
        "first": arguments.first,
        "second": arguments.second,
        "games": arguments.games,
        "base_ms": arguments.base_ms,
        "increment_ms": arguments.increment_ms,
        "ratings": {profile: round(rating, 1) for profile, rating in ratings.items()},
        "stats": stats,
        "terminations": dict(terminations),
    }
    print(json.dumps(result, indent=2))
    arguments.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
