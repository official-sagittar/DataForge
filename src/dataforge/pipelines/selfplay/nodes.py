from pathlib import Path
from .self_play import self_play


def generate_pgn_from_self_play(params: dict):
    self_play(
        tournament_runner_path=params["tournament_runner_path"],
        engine1_path=params["engine1_path"],
        engine2_path=params["engine2_path"],
        opening_book_path=params["opening_book_path"],
        opening_book_fmt=params["opening_book_fmt"],
        search_limit_type=params.get("search_limit_type", "depth"),
        tc=params.get("tc"),
        depth=params.get("depth", 10),
        nodes=params.get("nodes"),
        rounds=params["rounds"],
        output_dir=Path(params["output_dir"]),
        concurrency=params["concurrency"],
    )
