from kedro.pipeline import Pipeline, node
from .nodes import pgn_to_fen_with_wdl, sample_game_fens, convert_pos_to_quite


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=pgn_to_fen_with_wdl,
            inputs={
                "pgn_dir": "params:pgn_to_fen_with_wdl.pgn_dir",
                "output_dir": "params:pgn_to_fen_with_wdl.raw_fens_output_dir"
            },
            outputs="pgn_to_fen_with_wdl_path",
            name="pgn_to_fen_with_wdl_node"
        ),
        node(
            func=sample_game_fens,
            inputs={
                "raw_fens_csv_path": "pgn_to_fen_with_wdl_path",
                "output_dir": "params:pgn_to_fen_with_wdl.sampled_labelled_output_dir",
                "samples_per_game_pct": "params:pgn_to_fen_with_wdl.samples_per_game_pct"
            },
            outputs="sample_game_fens_path",
            name="sample_game_fens_node"
        ),
        node(
            func=convert_pos_to_quite,
            inputs={
                "raw_fens_csv_path": "sample_game_fens_path",
                "output_dir": "params:pgn_to_fen_with_wdl.quite_labelled_output_dir"
            },
            outputs="quiet_labelled_path",
            name="filter_quiet_positions_from_raw_node",
        ),
    ])
