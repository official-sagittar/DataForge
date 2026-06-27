from kedro.pipeline import Pipeline, node
from .nodes import pgn_to_fen_with_wdl


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=pgn_to_fen_with_wdl,
            inputs="params:pgn_to_fen_with_wdl.pgn_dir",
            outputs="raw_labelled_fens",
            name="pgn_to_fen_with_wdl_node"
        ),
    ])
