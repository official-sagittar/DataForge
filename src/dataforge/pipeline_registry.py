"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from dataforge.pipelines.selfplay import pipeline as selfplay_pipeline
from dataforge.pipelines.pgn_to_fen_with_wdl import pipeline as pgn_to_fen_with_wdl_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    pipelines["selfplay"] = selfplay_pipeline.create_pipeline()
    pipelines["pgn_to_fen_with_wdl"] = pgn_to_fen_with_wdl_pipeline.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
