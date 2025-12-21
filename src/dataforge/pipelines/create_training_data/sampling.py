import pandas as pd


def weighted_sample(
    df: pd.DataFrame,
    n_samples: int,
    n_phase_bins: int = 3,
    replace: bool = False,
    random_seed: int = 42,
    verify: bool = False,
) -> pd.DataFrame:
    """
    Perform weighted sampling to achieve an approximately uniform joint
    distribution over (phase, WDL).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns:
        - 'fen'   : str
        - 'phase' : int in [0, 256]
        - 'WDL'   : float in {0.0, 0.5, 1.0}

    n_samples : int
        Number of rows to sample.

    n_phase_bins : int, default=16
        Number of phase bins (16 or 32 recommended).

    replace : bool, default=False
        Sample with replacement if dataset is small.

    random_seed : int, default=42
        RNG seed for reproducibility.

    verify : bool, default=False
        If True, prints distribution diagnostics.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe with uniform (phase × WDL) expectation.
    """

    df = df.copy()

    # -----------------------------
    # VALIDATION
    # -----------------------------
    required_cols = {"fen", "phase", "wdl"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {required_cols}")

    if not df["phase"].between(0, 256).all():
        raise ValueError("Phase must be in range [0, 256]")

    if not df["wdl"].isin([0.0, 0.5, 1.0]).all():
        raise ValueError("WDL must be one of {0.0, 0.5, 1.0}")

    # -----------------------------
    # STEP 1: BIN PHASE
    # -----------------------------
    df["phase_bin"] = pd.cut(
        df["phase"],
        bins=n_phase_bins,
        labels=False,
        include_lowest=True,
    ).astype(int)

    df["wdl_bin"] = df["wdl"].astype(float)

    # -----------------------------
    # STEP 2: BIN COUNTS
    # -----------------------------
    bin_counts = (
        df
        .groupby(["phase_bin", "wdl_bin"])
        .size()
    )

    # -----------------------------
    # STEP 3: INVERSE-DENSITY WEIGHTS
    # -----------------------------
    inv_density = 1.0 / bin_counts
    inv_density /= inv_density.sum()

    df["weight"] = (
        df
        .set_index(["phase_bin", "wdl_bin"])
        .index
        .map(inv_density)
    )

    df = df[df["weight"].notna() & (df["weight"] > 0)]

    # -----------------------------
    # STEP 4: WEIGHTED SAMPLING
    # -----------------------------
    sampled = df.sample(
        n=n_samples,
        weights="weight",
        replace=replace,
        random_state=random_seed,
    ).reset_index(drop=True)

    # -----------------------------
    # OPTIONAL VERIFICATION
    # -----------------------------
    if verify:
        print("\nJoint distribution (phase_bin × WDL):")
        print(
            pd.crosstab(
                sampled["phase_bin"],
                sampled["wdl_bin"],
                normalize="all",
            ).round(4)
        )

        print("\nWDL marginal:")
        print(sampled["wdl"].value_counts(normalize=True))

        print("\nPhase marginal:")
        print(
            sampled["phase_bin"]
            .value_counts(normalize=True)
            .sort_index()
        )

    return sampled
