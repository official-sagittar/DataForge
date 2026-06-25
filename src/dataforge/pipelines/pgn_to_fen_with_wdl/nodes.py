import re
import uuid
from pathlib import Path
from datetime import datetime
import pandas as pd
import chess.pgn


def clamp(value, low, high):
    return max(low, min(value, high))


ENGINE_COMMENT_RE = re.compile(
    r"(?P<eval>[+-]?(?:\d+(?:\.\d+)?|\.\d+|M\d+)|#[+-]?\d+)"
    r"/(?P<depth>\d+)"
    r"\s+"
    r"(?P<time>\d+(?:\.\d+)?)s"
)


def parse_engine_comment(comment: str):
    match = ENGINE_COMMENT_RE.search(comment)
    if not match:
        return None, None, None, None

    raw_eval = match.group("eval")
    depth = int(match.group("depth"))
    engine_time = float(match.group("time"))

    if "M" in raw_eval:
        eval_type = "mate"
        eval_value = int(raw_eval.replace("+M", "").replace("M", "").replace("-M", "-"))
    elif raw_eval.startswith("#"):
        eval_type = "mate"
        eval_value = int(raw_eval[1:])
    else:
        eval_type = "cp"
        eval_value = float(raw_eval)

    return eval_type, eval_value, depth, engine_time


def calc_pos_phase(board) -> int:
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

    return clamp(int(phase), 0, 256)


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
                timectrl = game.headers.get("TimeControl", "-")
                termination = game.headers.get("Termination", "")
                duration = game.headers.get("GameDuration", "")
                plycnt = game.headers.get("PlyCount", 0)

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
                for node in game.mainline():
                    fen = board.fen()

                    phase = calc_pos_phase(board)
                    eval_type, eval_value, depth, engine_time = parse_engine_comment(node.comment)

                    pos_has_capture_moves = any(board.is_capture(m) for m in board.legal_moves)
                    is_quite_pos = (
                        pos_has_capture_moves == False
                        and board.is_check() == False
                        and eval_type != "mate"
                    )

                    rows.append((
                        game_id,
                        timectrl,
                        fen,
                        node.comment,
                        is_quite_pos,
                        phase,
                        eval_type,
                        eval_value,
                        depth,
                        engine_time,
                        wdl,
                        termination,
                        duration,
                        plycnt
                    ))

                    move = node.move
                    board.push(move)

    if not rows:
        raise ValueError(f"No valid positions found in {pgn_dir}")

    df = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "game_time_control",
            "pos_fen",
            "pos_pgn_comment",
            "pos_is_quiet",
            "pos_phase",
            "pos_eval_type",
            "pos_eval",
            "pos_depth",
            "pos_engine_time",
            "game_wdl",
            "game_termination",
            "game_duration",
            "game_plycount",
        ]
    )
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
