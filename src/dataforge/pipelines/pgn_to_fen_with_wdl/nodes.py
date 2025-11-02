import os
import pandas as pd
import chess.pgn
from pathlib import Path
from datetime import datetime


def pgn_to_fen_with_wdl(pgn_dir: str, output_dir: str) -> str:
    pgn_dir = Path(pgn_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pgn_dir = Path(pgn_dir)
    for pgn_file in pgn_dir.glob("*.pgn"):
        print(f"Processing {pgn_file.name}...")
        with open(pgn_file, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                result = game.headers.get("Result", "*")
                if result == "1-0":
                    wdl = 1
                elif result == "0-1":
                    wdl = 0
                elif result == "1/2-1/2":
                    wdl = 0.5
                else:
                    continue  # skip unfinished games

                board = game.board()
                for move in game.mainline_moves():
                    fen = board.fen()
                    rows.append((fen, wdl))
                    board.push(move)

    if not rows:
        raise ValueError(f"No valid positions found in {pgn_dir}")

    df = pd.DataFrame(rows, columns=["fen", "wdl"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Raw Labelled FENs_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    return str(output_path)


def select_quite_positions(raw_fens_csv_path: str, output_dir: str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_fens_csv_path)
    quiet_rows = []

    for _, row in df.iterrows():
        board = chess.Board(row["fen"])

        # Skip positions in check
        if board.is_check():
            continue

        # Skip positions which have a capture or promotion move as a legal move
        has_tactical_move = any(
            move.promotion or board.is_capture(move)
            for move in board.legal_moves
        )
        if has_tactical_move:
            continue

        quiet_rows.append(row)

    quiet_df = pd.DataFrame(quiet_rows)
    # Shuffle rows
    shuffled_df = quiet_df.sample(frac=1).reset_index(drop=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Quite Labelled_{timestamp}.csv"
    shuffled_df.to_csv(output_path, index=False)

    return output_path
