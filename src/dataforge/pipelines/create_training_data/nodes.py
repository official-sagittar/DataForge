import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def filter_quiet_positions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pos_is_quiet"]]


def remove_duplicate_positions(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=['pos_fen'], keep='first')


def remove_positions_from_short_games(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['game_plycount'] >= 20]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pos_stm"] = df["pos_fen"].str.split().str[1]
    df.loc[:, "pos_phase_label"] = np.where(
        df["pos_phase"] >= 128,
        "EG",
        "MG",
    )
    df["pos_eval_white_pov"] = np.where(
        df["pos_stm"] == "w",
        df["pos_eval_stm_pov"],
        -df["pos_eval_stm_pov"],
    )

    return df


def sample_positions_by_start_fen_phase(
    df: pd.DataFrame,
    sample_pct: float = 0.3,
    seed: int = 42,
    min_rows_per_group: int = 1,
    include_phase: bool = False,
) -> pd.DataFrame:
    """
    Samples positions group-wise.

    grouping_mode options:
      - "start_fen": samples within each game_start_fen group
      - "start_fen_phase": samples within each game_start_fen x pos_phase_label group

    Returns all original columns.
    """

    if not 0 < sample_pct <= 1:
        raise ValueError("sample_pct must be between 0 and 1")

    if min_rows_per_group < 0:
        raise ValueError("min_rows_per_group must be >= 0")

    group_cols = ["game_start_fen"]

    if include_phase:
        group_cols.append("pos_phase_label")


    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    def sample_group(group_df: pd.DataFrame) -> pd.Index:
        n = round(len(group_df) * sample_pct)
        n = max(min_rows_per_group, n)
        n = min(n, len(group_df))

        if n == 0:
            return pd.Index([])

        return group_df.sample(
            n=n,
            random_state=seed,
            replace=False,
        ).index

    sampled_indexes = (
        df
        .groupby(group_cols)
        .apply(sample_group, include_groups=False)
        .explode()
        .dropna()
        .to_list()
    )

    sampled_df = df.loc[sampled_indexes].reset_index(drop=True)

    return sampled_df


def remove_iqr_outliers_by_phase_wdl_stm(
    df: pd.DataFrame,
    include_stm: bool = False,
) -> pd.DataFrame:
    """
    Removes outliers using IQR bounds computed within:
        phase x WDL x stm

    Lower outliers:
        value < Q1 - (1.5 * IQR)

    Upper outliers:
        value > Q3 + (1.5 * IQR)
    """

    required_cols = ["pos_eval_white_pov", "pos_phase_label", "game_wdl", "pos_stm"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = df.copy()

    work_df["pos_eval_white_pov"] = pd.to_numeric(work_df["pos_eval_white_pov"], errors="coerce")
    work_df = work_df.dropna(subset=["pos_eval_white_pov"]).copy()

    group_cols = ["pos_phase_label", "game_wdl"]

    if include_stm:
        group_cols.append("pos_stm")

    iqr_bounds = (
        work_df
        .groupby(group_cols)["pos_eval_white_pov"]
        .agg(
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )

    iqr_bounds["iqr"] = iqr_bounds["q3"] - iqr_bounds["q1"]
    iqr_bounds["iqr_lower_bound"] = (
        iqr_bounds["q1"] - 1.5 * iqr_bounds["iqr"]
    )
    iqr_bounds["iqr_upper_bound"] = (
        iqr_bounds["q3"] + 1.5 * iqr_bounds["iqr"]
    )

    work_df = work_df.merge(
        iqr_bounds,
        on=group_cols,
        how="left",
    )

    work_df["is_lower_iqr_outlier"] = (
        work_df["pos_eval_white_pov"] < work_df["iqr_lower_bound"]
    )

    work_df["is_upper_iqr_outlier"] = (
        work_df["pos_eval_white_pov"] > work_df["iqr_upper_bound"]
    )

    work_df["is_iqr_outlier"] = (
        work_df["is_lower_iqr_outlier"]
        | work_df["is_upper_iqr_outlier"]
    )

    clean_df = work_df[~work_df["is_iqr_outlier"]].copy()
    return clean_df.reset_index(drop=True)


def tag_signal_noise_by_phase_wdl_stm_median(
    df: pd.DataFrame,
    draw_band: float = 0.05,
) -> pd.DataFrame:
    """
    Computes median eval at phase x WDL x stm level and tags rows as signal/noise.

    Signal if:
      - game_wdl == 1.0 and eval > group median
      - game_wdl == 0.0 and eval < group median
      - game_wdl == 0.5 and eval is within group median +/- draw_band

    Noise otherwise.
    """

    group_cols = ["pos_phase_label", "game_wdl", "pos_stm"]
    required_cols = group_cols + ["pos_eval_white_pov"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = df.copy()

    medians = (
        work_df
        .groupby(group_cols, dropna=False)["pos_eval_white_pov"]
        .median()
        .reset_index()
        .rename(columns={"pos_eval_white_pov": "group_median_eval_white_pov"})
    )

    work_df = work_df.merge(
        medians,
        on=group_cols,
        how="left",
    )

    white_win_signal = (
        (work_df["game_wdl"] == 1.0)
        & (work_df["pos_eval_white_pov"] > work_df["group_median_eval_white_pov"])
    )

    black_win_signal = (
        (work_df["game_wdl"] == 0.0)
        & (work_df["pos_eval_white_pov"] < work_df["group_median_eval_white_pov"])
    )

    draw_signal = (
        (work_df["game_wdl"] == 0.5)
        & work_df["pos_eval_white_pov"].between(
            work_df["group_median_eval_white_pov"] - draw_band,
            work_df["group_median_eval_white_pov"] + draw_band,
            inclusive="both",
        )
    )

    work_df["signal_type"] = "noise"
    work_df.loc[
        white_win_signal | black_win_signal | draw_signal,
        "signal_type",
    ] = "signal"

    return work_df.reset_index(drop=True)


def sample_uniform_phase_wdl_stm_signal_noise(
    df: pd.DataFrame,
    signal_ratio: float = 0.6,
    seed: int = 42,
    include_stm: bool = False,
) -> pd.DataFrame:
    """
    Samples the largest feasible balanced dataset without replacement.

    Guarantees:
      - if include_stm=True:
          uniform pos_phase_label x game_wdl x pos_stm
      - if include_stm=False:
          uniform pos_phase_label x game_wdl
      - within each stratum:
          signal_ratio signal rows
          1 - signal_ratio noise rows
      - no replacement

    The function automatically chooses the largest feasible strata_size.
    """

    if not 0 < signal_ratio < 1:
        raise ValueError("signal_ratio must be between 0 and 1")

    group_cols = ["pos_phase_label", "game_wdl"]

    if include_stm:
        group_cols.append("pos_stm")

    required_cols = group_cols + ["signal_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    availability = (
        df
        .groupby(group_cols + ["signal_type"], dropna=False)
        .size()
        .reset_index(name="available")
    )

    availability_pivot = (
        availability
        .pivot_table(
            index=group_cols,
            columns="signal_type",
            values="available",
            fill_value=0,
        )
        .reset_index()
    )

    if "signal" not in availability_pivot.columns:
        raise ValueError("No rows found with signal_type == 'signal'")

    if "noise" not in availability_pivot.columns:
        raise ValueError("No rows found with signal_type == 'noise'")

    availability_pivot["max_strata_from_signal"] = (
        availability_pivot["signal"] / signal_ratio
    )

    availability_pivot["max_strata_from_noise"] = (
        availability_pivot["noise"] / (1 - signal_ratio)
    )

    availability_pivot["max_feasible_strata_size"] = availability_pivot[
        ["max_strata_from_signal", "max_strata_from_noise"]
    ].min(axis=1)

    strata_size = math.floor(
        availability_pivot["max_feasible_strata_size"].min()
    )

    if strata_size <= 0:
        raise ValueError(
            "No feasible strata_size found. At least one stratum does not "
            "have enough signal/noise rows."
        )

    n_signal = math.floor(strata_size * signal_ratio)
    n_noise = strata_size - n_signal

    if n_signal <= 0 or n_noise <= 0:
        raise ValueError(
            f"Computed invalid sample sizes: "
            f"strata_size={strata_size}, n_signal={n_signal}, n_noise={n_noise}"
        )

    sampled_parts = []

    groups = list(df.groupby(group_cols, dropna=False))
    groups = sorted(groups, key=lambda x: str(x[0]))

    for i, (group_key, group_df) in enumerate(groups):
        signal_df = group_df[group_df["signal_type"] == "signal"]
        noise_df = group_df[group_df["signal_type"] == "noise"]

        sampled_signal = signal_df.sample(
            n=n_signal,
            replace=False,
            random_state=seed + i * 2,
        )

        sampled_noise = noise_df.sample(
            n=n_noise,
            replace=False,
            random_state=seed + i * 2 + 1,
        )

        sampled_parts.append(sampled_signal)
        sampled_parts.append(sampled_noise)

    sampled_df = (
        pd.concat(sampled_parts, ignore_index=True)
        .reset_index(drop=True)
    )

    return sampled_df


def shuffle_data(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Shuffles all rows in a DataFrame and resets the index.
    Keeps all original columns.
    """

    return (
        df
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def print_joint_distribution_phase_wdl_stm_signal(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prints joint distribution at:
        pos_phase_label x game_wdl x pos_stm x signal_type

    Expected combinations:
        2 phases x 3 WDL x 2 STM x 2 signal_type = 24 rows

    Returns the distribution as a DataFrame so Kedro can save/pass it.
    """

    group_cols = [
        "pos_phase_label",
        "game_wdl",
        "pos_stm",
        "signal_type",
    ]

    required_cols = group_cols
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    dist = (
        df
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    dist["pct"] = dist["count"] / dist["count"].sum() * 100

    expected_rows = 2 * 3 * 2 * 2
    actual_rows = len(dist)

    print("\nJoint distribution: phase x WDL x STM x signal_type")
    print("=" * 80)
    print(dist.to_string(index=False))
    print("=" * 80)

    if actual_rows != expected_rows:
        print(
            f"WARNING: Expected {expected_rows} combinations, "
            f"but found {actual_rows}. Some strata may be missing."
        )

    return dist


def print_eval_summary_by_phase_wdl_stm_signal(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prints count, mean, median, and std dev of eval_col at:
        pos_phase_label x game_wdl x pos_stm x signal_type

    Returns the summary DataFrame so Kedro can save it.
    """

    group_cols = [
        "pos_phase_label",
        "game_wdl",
        "pos_stm",
        "signal_type",
    ]

    required_cols = group_cols + ["pos_eval_white_pov"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = df.copy()

    summary = (
        work_df
        .groupby(group_cols, dropna=False)["pos_eval_white_pov"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std_dev="std",
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    print("\nEval summary: phase x WDL x STM x signal_type")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)
    print(f"Total rows summarized: {summary['count'].sum():,}")

    expected_rows = 2 * 3 * 2 * 2
    actual_rows = len(summary)

    if actual_rows != expected_rows:
        print(
            f"WARNING: Expected {expected_rows} groups, "
            f"but found {actual_rows}. Some combinations may be missing."
        )

    return summary


def write_epd(df: pd.DataFrame, output_dir: str) -> None:
    """
    Write training data to an EPD file.

    Format:
        <FEN> ; [WDL]

    Example:
        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ; [0.5]
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Training Data_{timestamp}.epd"

    with output_path.open("w", encoding="utf-8") as f:
        for fen, wdl in zip(df["pos_fen"], df["game_wdl"]):
            f.write(f"{fen} ; [{wdl}]\n")
