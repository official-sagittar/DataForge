import chess
import pandas as pd


def _calc_pos_phase(row) -> int:
    board = chess.Board(row['fen'])

    N = len(board.pieces(chess.KNIGHT, chess.WHITE))
    n = len(board.pieces(chess.KNIGHT, chess.BLACK))

    B = len(board.pieces(chess.BISHOP, chess.WHITE))
    b = len(board.pieces(chess.BISHOP, chess.BLACK))

    R = len(board.pieces(chess.ROOK, chess.WHITE))
    r = len(board.pieces(chess.ROOK, chess.BLACK))

    Q = len(board.pieces(chess.QUEEN, chess.WHITE))
    q = len(board.pieces(chess.QUEEN, chess.BLACK))

    phase = 24
    phase = phase - (N + n)
    phase = phase - (B + b)
    phase = phase - ((R + r) * 2)
    phase = phase - ((Q + q) * 4)
    phase = ((phase * 256 + (24 / 2)) / 24)

    return int(phase)


def create_features(df: pd.DataFrame) -> None:
    df['phase'] = df.apply(_calc_pos_phase, axis=1)
