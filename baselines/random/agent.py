import random

import chess


def get_move(fen: str, time_left_ms: int) -> str:
    return random.choice(list(chess.Board(fen).legal_moves)).uci()
