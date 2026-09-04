from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent  # noqa: E402


class AgentSmokeTests(unittest.TestCase):
    def assert_legal_and_fast(self, fen: str) -> None:
        board = chess.Board(fen)
        started = time.perf_counter()
        move = chess.Move.from_uci(agent.get_move(fen, 1_000))
        elapsed = time.perf_counter() - started
        self.assertIn(move, board.legal_moves)
        self.assertLess(elapsed, 1.5)

    def test_starting_position(self) -> None:
        self.assert_legal_and_fast(chess.STARTING_FEN)

    def test_time_budget_protects_low_clock(self) -> None:
        board = chess.Board()
        moves = list(board.legal_moves)
        low_budget = agent.allocate_time(board, 2_000, moves)
        full_budget = agent.allocate_time(board, 120_000, moves)
        self.assertLess(low_budget, full_budget)
        self.assertLessEqual(agent.allocate_time(board, 2_000, moves), 250)

    def test_time_profiles_are_ordered(self) -> None:
        board = chess.Board()
        moves = list(board.legal_moves)
        conservative = agent.allocate_time(board, 120_000, moves, "conservative")
        balanced = agent.allocate_time(board, 120_000, moves, "balanced")
        tactical = agent.allocate_time(board, 120_000, moves, "tactical")
        self.assertLess(conservative, balanced)
        self.assertGreaterEqual(tactical, balanced)

    def test_promotion(self) -> None:
        self.assert_legal_and_fast("7k/P7/7K/8/8/8/8/8 w - - 0 1")

    def test_en_passant(self) -> None:
        self.assert_legal_and_fast("8/8/8/3pP3/8/8/8/4K2k w - d6 0 2")

    def test_checkmate_has_no_move(self) -> None:
        fen = "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"
        with self.assertRaises(ValueError):
            agent.get_move(fen, 1_000)
