from pathlib import Path
from datetime import datetime
import uuid
import pandas as pd
import chess.pgn


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

                game_id = str(uuid.uuid4())

                board = game.board()
                for move in game.mainline_moves():
                    fen = board.fen()
                    rows.append((game_id, fen, wdl))
                    board.push(move)

    if not rows:
        raise ValueError(f"No valid positions found in {pgn_dir}")

    df = pd.DataFrame(rows, columns=["game_id", "fen", "wdl"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Raw Labelled FENs_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    return str(output_path)


def sample_game_fens(raw_fens_csv_path: str, output_dir: str, samples_per_game_pct: float = 0.10, seed: int = 42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_fens_df = pd.read_csv(raw_fens_csv_path)

    sampled_raw_fens_df = raw_fens_df.groupby("game_id", group_keys=False).sample(frac=samples_per_game_pct, random_state=seed).reset_index(drop=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Sampled Labelled FENs_{timestamp}.csv"
    sampled_raw_fens_df.to_csv(output_path, index=False)

    return output_path


def convert_pos_to_quite(raw_fens_csv_path: str, output_dir: str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_fens_csv_path)
    quiet_rows = []

    for index, row in df.iterrows():
        board = chess.Board(row["fen"])

        if board.is_check():
            continue

        has_capture_moves = any(board.is_capture(m) for m in board.legal_moves)

        if not has_capture_moves:
            quiet_rows.append(row)

    quiet_df = pd.DataFrame(quiet_rows)
    quiet_df = quiet_df[["fen", "wdl"]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Quite Labelled_{timestamp}.csv"
    quiet_df.to_csv(output_path, index=False)

    return output_path
