from kedro.pipeline import Pipeline, node
from .nodes import create_training_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_training_data,
            inputs={
                "quite_labelled_data_path": "params:create_training_data.quite_labelled_dir",
                "output_dir": "params:create_training_data.training_data_output_dir",
                "size": "params:create_training_data.size"
            },
            outputs="training_data_path",
            name="create_training_data",
        ),
    ])
