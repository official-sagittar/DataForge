from kedro.pipeline import Pipeline, node
from .nodes import create_opening_book

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_opening_book,
            inputs={
                "seed_openings": "params:create_opening_book.seed_openings",
                "output_dir": "params:create_opening_book.output_dir",
                "size": "params:create_opening_book.size",
                "random_seed": "params:create_opening_book.random_seed",
            },
            outputs="opening_book_path",
            name="create_opening_book"
        )
    ])
