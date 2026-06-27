from kedro.pipeline import Pipeline, node
from .nodes import generate_pgn_from_self_play


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=generate_pgn_from_self_play,
            inputs="params:self_play",
            outputs=None,
            name="self_play_node"
        )
    ])
