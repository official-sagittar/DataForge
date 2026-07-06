from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=filter_quiet_positions,
            inputs="raw_labelled_fens",
            outputs="quiet_positions",
            name="filter_quiet_positions_node",
        ),
        node(
            func=combine_partitions,
            inputs="quiet_positions",
            outputs="quiet_positions_combined",
            name="combine_partitions_node",
        ),
        node(
            func=remove_duplicate_positions,
            inputs="quiet_positions_combined",
            outputs="dedup_quiet_positions",
            name="remove_duplicate_positions_node",
        ),
        node(
            func=remove_positions_from_short_games,
            inputs="dedup_quiet_positions",
            outputs="qualified_quiet_positions",
            name="remove_positions_from_short_games_node",
        ),
        node(
            func=remove_positions_from_early_ply,
            inputs="qualified_quiet_positions",
            outputs="qualified_quiet_positions_without_early_plys",
            name="remove_positions_from_early_ply_node",
        ),
        node(
            func=add_features,
            inputs="qualified_quiet_positions_without_early_plys",
            outputs="qualified_quiet_positions_with_features",
            name="add_features_node",
        ),
        node(
            func=sample_positions_by_start_fen_phase,
            inputs="qualified_quiet_positions_with_features",
            outputs="sampled_data",
            name="sample_positions_by_start_fen_phase_node",
        ),
        node(
            func=remove_iqr_outliers_by_phase_wdl_stm,
            inputs="sampled_data",
            outputs="sampled_data_with_no_outliers",
            name="remove_iqr_outliers_by_phase_wdl_stm_node",
        ),
        node(
            func=tag_signal_noise_by_phase_wdl_stm_median,
            inputs="sampled_data_with_no_outliers",
            outputs="signal_tagged_data",
            name="tag_signal_noise_by_phase_wdl_stm_median_node",
        ),
        node(
            func=sample_uniform_phase_wdl_stm_signal_noise,
            inputs="signal_tagged_data",
            outputs="sampled_training_data",
            name="sample_uniform_phase_wdl_stm_signal_noise_node",
        ),
        node(
            func=shuffle_data,
            inputs="sampled_training_data",
            outputs="training_data",
            name="shuffle_data_node",
        ),
        node(
            func=print_joint_distribution_phase_wdl_stm_signal,
            inputs="training_data",
            outputs=None,
            name="print_joint_distribution_phase_wdl_stm_signal_node",
        ),
        node(
            func=print_eval_summary_by_phase_wdl_stm_signal,
            inputs="training_data",
            outputs=None,
            name="print_eval_summary_by_phase_wdl_stm_signal_node",
        ),
        node(
            func=plot_eval_boxplot_phase_wdl_stm_signal,
            inputs={
                "df": "training_data",
                "output_dir": "params:create_training_data.training_data_output_dir",
            },
            outputs=None,
            name="plot_eval_boxplot_phase_wdl_stm_signal_node",
        ),
        node(
            func=write_epd,
            inputs={
                "df": "training_data",
                "output_dir": "params:create_training_data.training_data_output_dir",
            },
            outputs=None,
            name="write_epd_node",
        ),
    ])
