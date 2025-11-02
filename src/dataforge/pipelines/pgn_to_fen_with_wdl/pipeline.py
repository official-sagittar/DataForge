from kedro.pipeline import Pipeline, node
from .nodes import pgn_to_fen_with_wdl, select_quite_positions


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=pgn_to_fen_with_wdl,
            inputs={
                "pgn_dir": "params:pgn_to_fen_with_wdl.pgn_dir",
                "output_dir": "params:pgn_to_fen_with_wdl.output_dir"
            },
            outputs="pgn_to_fen_with_wdl_path",
            name="pgn_to_fen_with_wdl_node"
        ),
        node(
            func=select_quite_positions,
            inputs={
                "raw_fens_csv_path": "pgn_to_fen_with_wdl_path",
                "output_dir": "params:pgn_to_fen_with_wdl.quite_labelled_output_dir"
            },
            outputs="quiet_labelled_path",
            name="filter_quiet_positions_from_raw_node",
        ),
    ])
