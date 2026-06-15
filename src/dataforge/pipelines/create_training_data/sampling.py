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


def sample_balance_wdl(
    df: pd.DataFrame,
    n_samples: int,
    replace: bool = False,
    random_seed: int = 42,
    verify: bool = False,
) -> pd.DataFrame:
    """
    Sample rows such that the sampled dataset has (approximately) equal counts
    for each WDL class {0.0, 0.5, 1.0}, without enforcing any phase balancing.

    Expected target: ~n_samples/3 rows per WDL class.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns:
        - 'fen'   : str
        - 'phase' : int in [0, 256]
        - 'wdl'   : float in {0.0, 0.5, 1.0}

    n_samples : int
        Number of rows to sample.

    replace : bool, default=False
        Sample with replacement if some WDL class has too few rows.

    random_seed : int, default=42
        RNG seed for reproducibility.

    verify : bool, default=False
        If True, prints distribution diagnostics.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe with ~uniform WDL marginal.
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
    # STEP 1: SPLIT BY WDL
    # -----------------------------
    classes = [0.0, 0.5, 1.0]
    groups = {c: df[df["wdl"] == c] for c in classes}

    # -----------------------------
    # STEP 2: DECIDE PER-CLASS QUOTAS
    # -----------------------------
    base = n_samples // 3
    rem = n_samples - 3 * base
    # deterministic remainder distribution (so runs are stable)
    quotas = {0.0: base, 0.5: base, 1.0: base}
    for c in classes[:rem]:
        quotas[c] += 1

    # If not replacing, cap quotas to what's available per class
    if not replace:
        for c in classes:
            quotas[c] = min(quotas[c], len(groups[c]))

        # If we capped, we may end up short; fill the remainder from the whole df
        got = sum(quotas.values())
        deficit = n_samples - got
    else:
        deficit = 0

    # -----------------------------
    # STEP 3: SAMPLE PER CLASS
    # -----------------------------
    sampled_parts = []
    for c in classes:
        part = groups[c].sample(
            n=quotas[c],
            replace=replace,
            random_state=(random_seed + int(c * 10) + 1),
        )
        sampled_parts.append(part)

    sampled = pd.concat(sampled_parts, axis=0)

    # -----------------------------
    # STEP 4: IF NEEDED, TOP UP (ONLY WHEN replace=False AND CLASS SHORTAGE)
    # -----------------------------
    if deficit > 0:
        # sample remaining rows from the rest (excluding already chosen indices)
        remaining = df.drop(index=sampled.index, errors="ignore")
        if len(remaining) == 0:
            # nothing left; fall back to sampling with replacement from full df
            topup = df.sample(
                n=deficit, replace=True, random_state=random_seed + 999
            )
        else:
            topup = remaining.sample(
                n=min(deficit, len(remaining)),
                replace=False,
                random_state=random_seed + 999,
            )

        sampled = pd.concat([sampled, topup], axis=0)

    # -----------------------------
    # OPTIONAL VERIFICATION
    # -----------------------------
    if verify:
        print("\nWDL marginal:")
        print(sampled["wdl"].value_counts(normalize=True).sort_index().round(4))

        # Optional: show phase distribution but don't constrain it
        print("\nPhase (raw) summary:")
        print(sampled["phase"].describe())

    return sampled


def sample_wdl_x_stm_uniform(
    df: pd.DataFrame,
    n_samples: int,
    random_seed: int = 42,
    verify: bool = False,
) -> pd.DataFrame:
    """
    Sample WITHOUT replacement to be as-uniform-as-possible over:
      wdl ∈ {0.0, 0.5, 1.0}  ×  stm ∈ {'w', 'b'}

    Assumes df already has columns: 'fen', 'wdl', 'stm'.
    """

    df = df.copy()

    required_cols = {"fen", "wdl", "stm"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {required_cols}")

    if not df["wdl"].isin([0.0, 0.5, 1.0]).all():
        bad = df.loc[~df["wdl"].isin([0.0, 0.5, 1.0]), "wdl"].unique()
        raise ValueError(f"wdl must be one of {{0.0, 0.5, 1.0}}. Found: {bad}")

    # normalize stm to 'w'/'b'
    df["stm"] = df["stm"].astype(str).str.strip().str.lower()
    if not df["stm"].isin(["w", "b"]).all():
        bad = df.loc[~df["stm"].isin(["w", "b"]), "stm"].unique()[:10]
        raise ValueError(f"stm must be 'w' or 'b'. Examples of bad values: {bad}")

    bucket_order = [(0.0, "w"), (0.0, "b"), (0.5, "w"), (0.5, "b"), (1.0, "w"), (1.0, "b")]
    buckets = {(w, s): df[(df["wdl"] == w) & (df["stm"] == s)] for (w, s) in bucket_order}

    total_available = sum(len(b) for b in buckets.values())
    if total_available == 0:
        return df.iloc[0:0].copy()

    target_total = min(n_samples, total_available)

    base = target_total // 6
    rem = target_total - 6 * base

    targets = {k: base for k in bucket_order}
    for k in bucket_order[:rem]:
        targets[k] += 1

    # cap + redistribute (still no replacement)
    supply = {k: len(buckets[k]) for k in bucket_order}
    assigned = {k: min(targets[k], supply[k]) for k in bucket_order}

    leftover = target_total - sum(assigned.values())
    while leftover > 0:
        progressed = False
        for k in bucket_order:
            if leftover == 0:
                break
            if assigned[k] < supply[k]:
                assigned[k] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break

    parts = []
    for (w, s) in bucket_order:
        n_k = assigned[(w, s)]
        if n_k <= 0:
            continue
        rs = random_seed + int(w * 10) + (0 if s == "w" else 1) + 123
        parts.append(buckets[(w, s)].sample(n=n_k, replace=False, random_state=rs))

    sampled = pd.concat(parts, axis=0)

    if verify:
        out = sampled.copy()
        out["bucket"] = out["wdl"].astype(str) + "_" + out["stm"]
        counts = out["bucket"].value_counts().reindex(
            [f"{w}_{s}" for (w, s) in bucket_order], fill_value=0
        )
        print("\nBucket counts (wdl_stm):")
        print(counts)
        print("\nWDL marginal:")
        print(out["wdl"].value_counts(normalize=True).sort_index().round(4))
        print("\nSTM marginal:")
        print(out["stm"].value_counts(normalize=True).reindex(["w", "b"]).round(4))
        if len(sampled) < n_samples:
            print(f"\nNote: requested {n_samples}, returned {len(sampled)} (no replacement).")

    return sampled
