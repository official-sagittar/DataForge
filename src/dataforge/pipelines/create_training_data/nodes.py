import math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def filter_quiet_positions(df: pd.DataFrame) -> pd.DataFrame:
    print(df.shape)
    return df[df["pos_is_quiet"]]


def remove_duplicate_positions(df: pd.DataFrame) -> pd.DataFrame:
    print(df.shape)
    return df.drop_duplicates(subset=['pos_fen'], keep='first')


def remove_positions_from_short_games(df: pd.DataFrame) -> pd.DataFrame:
    print(df.shape)
    return df[df['game_plycount'] >= 20]


def remove_positions_from_early_ply(df: pd.DataFrame) -> pd.DataFrame:
    print(df.shape)
    return df[df['pos_ply'] > 8]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print(df.shape)
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
    sample_pct: float = 0.5,
    seed: int = 42,
    min_rows_per_group: int = 1,
    include_phase: bool = True,
) -> pd.DataFrame:
    """
    Samples positions group-wise.

    grouping_mode options:
      - "start_fen": samples within each game_start_fen group
      - "start_fen_phase": samples within each game_start_fen x pos_phase_label group

    Returns all original columns.
    """

    print(df.shape)

    if not 0 < sample_pct <= 1:
        raise ValueError("sample_pct must be between 0 and 1")

    if sample_pct ==  1.0:
        return df

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
    include_stm: bool = True,
) -> pd.DataFrame:
    """
    Removes outliers using IQR bounds computed within:
        phase x WDL x stm

    Lower outliers:
        value < Q1 - (1.5 * IQR)

    Upper outliers:
        value > Q3 + (1.5 * IQR)
    """

    print(df.shape)

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

    print(clean_df.shape)

    return clean_df.reset_index(drop=True)


def tag_signal_noise_by_phase_wdl_stm_median(
    df: pd.DataFrame,
    draw_band: float = 0.15,
    include_stm = True
) -> pd.DataFrame:
    """
    Computes median eval at phase x WDL x stm level and tags rows as signal/noise.

    Signal if:
      - game_wdl == 1.0 and eval > group median
      - game_wdl == 0.0 and eval < group median
      - game_wdl == 0.5 and eval is within group median +/- draw_band

    Noise otherwise.
    """

    group_cols = ["pos_phase_label", "game_wdl"]
    if include_stm:
        group_cols.append("pos_stm")

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
    signal_ratio: float = 0.7,
    seed: int = 42,
    include_stm: bool = True,
    include_draws_in_strata_size: bool = False,
) -> pd.DataFrame:
    """
    Samples a balanced dataset without replacement.

    Guarantees:
      - if include_stm=True:
          groups by pos_phase_label x game_wdl x pos_stm
      - if include_stm=False:
          groups by pos_phase_label x game_wdl
      - within each sampled stratum:
          approximately signal_ratio signal rows
          approximately 1 - signal_ratio noise rows
      - no replacement

    include_draws_in_strata_size:
      - True:
          current behavior. Draws are included when computing global strata_size.
          All strata get the same size.

      - False:
          strata_size is computed only from decisive games: game_wdl 0.0 and 1.0.
          Decisive strata get that full strata_size.
          Draw strata are capped at their own feasible size if smaller.
          This prevents weak draw buckets from bottlenecking the whole dataset.
    """

    print(df.shape)

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

    print(availability_pivot)

    if include_draws_in_strata_size:
        strata_size_source = availability_pivot
    else:
        strata_size_source = availability_pivot[
            availability_pivot["game_wdl"].isin([0.0, 1.0])
        ]

        if strata_size_source.empty:
            raise ValueError(
                "No decisive-game strata found. "
                "Expected game_wdl values 0.0 and/or 1.0."
            )

    global_strata_size = math.floor(
        strata_size_source["max_feasible_strata_size"].min()
    )

    if global_strata_size <= 0:
        raise ValueError(
            "No feasible strata_size found. At least one selected stratum "
            "does not have enough signal/noise rows."
        )

    print(f"Global strata_size: {global_strata_size:,}")
    print(f"include_draws_in_strata_size: {include_draws_in_strata_size}")

    sampled_parts = []

    groups = list(df.groupby(group_cols, dropna=False))
    groups = sorted(groups, key=lambda x: str(x[0]))

    for i, (group_key, group_df) in enumerate(groups):
        # group_key can be scalar or tuple depending on group_cols length
        if len(group_cols) == 1:
            group_key_values = {group_cols[0]: group_key}
        else:
            group_key_values = dict(zip(group_cols, group_key))

        game_wdl = group_key_values["game_wdl"]

        availability_row = availability_pivot.copy()

        for col, value in group_key_values.items():
            availability_row = availability_row[availability_row[col] == value]

        if availability_row.empty:
            raise ValueError(f"Could not find availability row for group {group_key}")

        max_feasible_for_group = math.floor(
            availability_row["max_feasible_strata_size"].iloc[0]
        )

        if include_draws_in_strata_size:
            target_strata_size = global_strata_size
        else:
            if game_wdl == 0.5:
                # Draws are allowed to be smaller.
                target_strata_size = min(
                    global_strata_size,
                    max_feasible_for_group,
                )
            else:
                # Decisive games must hit the global size.
                target_strata_size = global_strata_size

        n_signal = math.floor(target_strata_size * signal_ratio)
        n_noise = target_strata_size - n_signal

        if n_signal <= 0 or n_noise <= 0:
            raise ValueError(
                f"Computed invalid sample sizes for group {group_key}: "
                f"target_strata_size={target_strata_size}, "
                f"n_signal={n_signal}, n_noise={n_noise}"
            )

        signal_df = group_df[group_df["signal_type"] == "signal"]
        noise_df = group_df[group_df["signal_type"] == "noise"]

        if len(signal_df) < n_signal or len(noise_df) < n_noise:
            raise ValueError(
                f"Not enough rows for group {group_key}: "
                f"need {n_signal} signal / {n_noise} noise, "
                f"available {len(signal_df)} signal / {len(noise_df)} noise"
            )

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

        print(
            f"group={group_key} | "
            f"target={target_strata_size:,} | "
            f"signal={n_signal:,} | noise={n_noise:,}"
        )

    sampled_df = (
        pd.concat(sampled_parts, ignore_index=True)
        .reset_index(drop=True)
    )

    print(sampled_df.shape)

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


def plot_eval_boxplot_phase_wdl_stm_signal(
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Saves a box plot of eval distribution at:
        pos_phase_label x pos_stm x signal_type

    Split/color-coded by game_wdl:
      - game_wdl == 0.0: black
      - game_wdl == 0.5: light gray
      - game_wdl == 1.0: white

    Uses actual position-level rows, not the aggregated summary.
    """

    eval_col = "pos_eval_white_pov"

    required_cols = [
        "pos_phase_label",
        "game_wdl",
        "pos_stm",
        "signal_type",
        eval_col,
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    plot_df = df.copy()

    plot_df[eval_col] = pd.to_numeric(
        plot_df[eval_col],
        errors="coerce",
    )
    plot_df = plot_df.dropna(subset=[eval_col]).copy()

    plot_df["game_wdl_label"] = plot_df["game_wdl"].map(
        {
            0.0: "Black win",
            0.5: "Draw",
            1.0: "White win",
        }
    )

    if plot_df["game_wdl_label"].isna().any():
        bad_values = plot_df.loc[
            plot_df["game_wdl_label"].isna(),
            "game_wdl",
        ].unique()

        raise ValueError(f"Unexpected game_wdl values found: {bad_values}")

    plot_df["plot_group"] = (
        plot_df["pos_phase_label"].astype(str)
        + " | "
        + plot_df["pos_stm"].astype(str)
        + " | "
        + plot_df["signal_type"].astype(str)
    )

    group_order = [
        "MG | w | signal",
        "MG | w | noise",
        "MG | b | signal",
        "MG | b | noise",
        "EG | w | signal",
        "EG | w | noise",
        "EG | b | signal",
        "EG | b | noise",
    ]

    group_order = [
        group for group in group_order
        if group in set(plot_df["plot_group"])
    ]

    hue_order = ["Black win", "Draw", "White win"]

    palette = {
        "Black win": "black",
        "Draw": "lightgray",
        "White win": "white",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Training Data Eval Plot_{timestamp}.svg"

    plt.figure(figsize=(18, 10))

    ax = sns.boxplot(
        data=plot_df,
        x="plot_group",
        y=eval_col,
        hue="game_wdl_label",
        order=group_order,
        hue_order=hue_order,
        palette=palette,
        showfliers=False,
        linewidth=1.0,
    )

    # Add median labels
    medians = (
        plot_df
        .groupby(["plot_group", "game_wdl_label"], dropna=False)[eval_col]
        .median()
        .reset_index()
    )

    x_positions = {group: i for i, group in enumerate(group_order)}
    hue_offsets = {
        "Black win": -0.267,
        "Draw": 0.000,
        "White win": 0.267,
    }

    for _, row in medians.iterrows():
        group = row["plot_group"]
        wdl_label = row["game_wdl_label"]
        median_value = row[eval_col]

        if group not in x_positions or wdl_label not in hue_offsets:
            continue

        x = x_positions[group] + hue_offsets[wdl_label]
        y = median_value

        ax.text(
            x,
            y,
            f"{median_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="red",
        )

    # Add count and dataset percentage labels
    total_rows = len(plot_df)

    dist = (
        plot_df
        .groupby(["plot_group", "game_wdl_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    dist["pct"] = dist["count"] / total_rows * 100

    y_min = plot_df[eval_col].min()
    y_max = plot_df[eval_col].max()
    y_range = y_max - y_min

    label_y = y_max + 0.03 * y_range

    for _, row in dist.iterrows():
        group = row["plot_group"]
        wdl_label = row["game_wdl_label"]
        count = int(row["count"])
        pct = row["pct"]

        if group not in x_positions or wdl_label not in hue_offsets:
            continue

        x = x_positions[group] + hue_offsets[wdl_label]

        ax.text(
            x,
            label_y,
            f"n={count:,}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90,
            color="black",
        )

    # Add total dataset rows label in top-right
    ax.text(
        0.99,
        0.98,
        f"Total rows: {total_rows:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.85,
        },
    )

    # Give top labels some headroom
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.20 * y_range)

    # Make white and light-gray boxes visible.
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.6,
    )

    ax.set_title(
        "Eval Distribution by Phase x STM x Signal Type, split by Game Result"
    )
    ax.set_xlabel("Phase | Side to Move | Signal Type")
    ax.set_ylabel(eval_col)

    plt.xticks(rotation=90, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    clean = dict(zip(labels[:3], handles[:3]))

    ax.legend(
        clean.values(),
        clean.keys(),
        title="Game result",
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()


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
