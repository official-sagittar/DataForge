import re
import uuid
from pathlib import Path
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


def is_quiet_pos(board, best_move, eval_type) -> bool:
    if board.is_check():
        return False

    if board.is_capture(best_move):
        return False

    if eval_type == "mate":
        return False

    return not any(board.is_capture(m) for m in board.legal_moves)


def pgn_to_fen_with_wdl(pgn_dir: str) ->  dict[str, pd.DataFrame]:
    pgn_dir = Path(pgn_dir)

    columns=[
        "game_id",
        "game_time_control",
        "game_start_fen",
        "pos_ply",
        "pos_fen",
        "pos_bestmove",
        "pos_pgn_comment",
        "pos_is_quiet",
        "pos_phase",
        "pos_eval_type",
        "pos_eval_stm_pov",
        "pos_depth",
        "pos_engine_time",
        "game_wdl",
        "game_termination",
        "game_duration",
        "game_plycount",
    ]

    partitions = {}

    pgn_dir = Path(pgn_dir)
    for pgn_file in pgn_dir.glob("*.pgn"):
        print(f"Processing {pgn_file.name}...")
        file_rows = []

        with open(pgn_file, "r", encoding="utf-8") as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                except:
                    print("Could not read game!")
                    continue

                if game is None:
                    break

                result = game.headers.get("Result", "*")
                starting_fen = game.headers.get("FEN", "")
                timectrl = game.headers.get("TimeControl", "-")
                termination = game.headers.get("Termination", "")
                duration = game.headers.get("GameDuration", "")
                plycnt = int(game.headers.get("PlyCount", 0))

                if termination == "illegal move":
                    continue

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
                ply = 0
                for node in game.mainline():
                    fen = board.fen()
                    move = node.move

                    phase = calc_pos_phase(board)
                    eval_type, eval_value, depth, engine_time = parse_engine_comment(node.comment)
                    is_quiet = is_quiet_pos(board, move, eval_type)

                    file_rows.append((
                        game_id,
                        timectrl,
                        starting_fen,
                        ply,
                        fen,
                        move.uci(),
                        node.comment,
                        is_quiet,
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

                    board.push(move)
                    ply = ply + 1

        if file_rows:
            file_df = pd.DataFrame(file_rows, columns=columns)
            partitions[pgn_file.stem] = file_df

    return partitions
