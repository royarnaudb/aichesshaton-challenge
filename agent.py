"""A small, time-safe chess search for the AI Chessathon starter."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import chess

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
MATE_SCORE = 1_000_000
INF = MATE_SCORE + 1
MAX_QUIET_DEPTH = 8
MAX_MOVE_BUDGET_MS = 6_000
SAFETY_BUFFER_MS = 100


class SearchTimeout(Exception):
    """Raised when a completed iterative-deepening result is needed immediately."""


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    move: chess.Move | None


def allocate_time(
    board: chess.Board,
    time_left_ms: int,
    legal_moves: list[chess.Move],
    profile: str | None = None,
) -> int:
    """Choose a move budget while protecting the clock for future positions."""
    profile = profile or os.environ.get("CHESS_TIME_PROFILE", "balanced")
    if time_left_ms <= 500:
        return max(50, time_left_ms // 3)

    # Keep a reserve because the referee measures wall time and the increment arrives
    # only after the move is returned.
    reserve_ms = max(250, min(2_000, time_left_ms // 20))
    usable_ms = max(100, time_left_ms - reserve_ms)
    moves_to_go = 30 if board.fullmove_number < 20 else 22
    if len(board.piece_map()) <= 10:
        moves_to_go = 12 if profile == "endgame" else 16
    budget_ms = usable_ms // moves_to_go + 500

    captures = sum(board.is_capture(move) for move in legal_moves)
    if board.is_check():
        budget_ms = int(budget_ms * 1.8)
    elif captures >= 3:
        budget_ms = int(budget_ms * 1.25)
    if len(legal_moves) >= 30:
        budget_ms = int(budget_ms * 1.15)

    if profile == "conservative":
        budget_ms = int(budget_ms * 0.70)
    elif profile == "tactical":
        if board.is_check():
            budget_ms = int(budget_ms * 1.25)
        elif captures >= 3:
            budget_ms = int(budget_ms * 1.20)

    # Once the clock is low, consistency matters more than finding one extra ply.
    if time_left_ms < 3_000:
        budget_ms = min(budget_ms, max(75, time_left_ms // 8))
    return min(MAX_MOVE_BUDGET_MS, max(75, budget_ms))


class Search:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0
        self.table: dict[object, TTEntry] = {}

    def check_time(self) -> None:
        self.nodes += 1
        if self.nodes & 63 == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def evaluate(self, board: chess.Board) -> int:
        """Evaluate from the side-to-move perspective."""
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        mover = board.turn
        score = sum(
            value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
            for piece, value in PIECE_VALUE.items()
        )
        return score + 4 * board.legal_moves.count()

    def move_order(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None
    ) -> list[chess.Move]:
        def priority(move: chess.Move) -> tuple[int, int, int]:
            capture_value = 0
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                victim = PIECE_VALUE.get(captured.piece_type, 0) if captured else 100
                attacker = board.piece_at(move.from_square)
                capture_value = (
                    10 * victim - PIECE_VALUE.get(attacker.piece_type, 0)
                    if attacker
                    else victim
                )
            return (int(move == tt_move), capture_value, PIECE_VALUE.get(move.promotion or 0, 0))

        return sorted(moves, key=priority, reverse=True)

    def quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int, quiet_depth: int
    ) -> int:
        self.check_time()
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if board.is_check() else 0
        if quiet_depth >= MAX_QUIET_DEPTH:
            return self.evaluate(board)
        if not board.is_check():
            stand_pat = self.evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            moves = [move for move in moves if board.is_capture(move) or move.promotion]
            if not moves:
                return alpha
        for move in self.move_order(board, moves, None):
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, quiet_depth + 1)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.check_time()
        key = board._transposition_key()
        entry = self.table.get(key)
        if entry is not None and entry.depth >= depth:
            return entry.score
        if depth == 0:
            return self.quiescence(board, alpha, beta, ply, 0)
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if board.is_check() else 0
        best = -INF
        best_move: chess.Move | None = None
        tt_move = entry.move if entry is not None else None
        cutoff = False
        for move in self.move_order(board, moves, tt_move):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                cutoff = True
                break
        if not cutoff:
            self.table[key] = TTEntry(depth, best, best_move)
        return best

    def root(self, board: chess.Board, depth: int, previous: chess.Move) -> chess.Move:
        moves = list(board.legal_moves)
        best_move = previous if previous in moves else moves[0]
        best_score = -INF
        entry = self.table.get(board._transposition_key())
        tt_move = entry.move if entry is not None else best_move
        for move in self.move_order(board, moves, tt_move):
            board.push(move)
            score = -self.negamax(board, depth - 1, -INF, INF, 1)
            board.pop()
            if score > best_score:
                best_score, best_move = score, move
        return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("position has no legal moves")
    budget_ms = allocate_time(board, time_left_ms, legal_moves)
    started_at = time.perf_counter()
    deadline = started_at + max(1, budget_ms - SAFETY_BUFFER_MS) / 1_000
    search = Search(deadline)
    best_move = legal_moves[0]
    depth = 1
    completed_depth = 0
    while depth <= 64:
        try:
            best_move = search.root(board, depth, best_move)
        except SearchTimeout:
            break
        completed_depth = depth
        depth += 1
    print(
        "METRIC "
        + json.dumps(
            {
                "nodes": search.nodes,
                "depth": completed_depth,
                "budget_ms": budget_ms,
                "time_used_ms": round((time.perf_counter() - started_at) * 1000),
            },
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    return best_move.uci()
