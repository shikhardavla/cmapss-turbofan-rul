import pandas as pd


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for each engine unit at each time
    cycle, serving as the primary label for this project.

    RUL = time_cycles_max - time_cycles_current

    Note: this assumes each unit's trajectory runs to failure (true for the
    CMAPSS training set). Do not apply this to the test set — test
    trajectories are truncated before failure, so RUL there must come from
    the provided RUL_*.txt ground truth instead.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit_number' and 'time_cycles'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'RUL' column.
    """
    df = df.copy()

    max_cycles = (
        df.groupby("unit_number")["time_cycles"]
        .transform("max")
    )

    df["RUL"] = max_cycles - df["time_cycles"]

    return df


def clip_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    RUL curves typically have a 'plateau' zone (engine healthy) followed by
    a rapid drop-off zone (engine degrading). We clip RUL at this 'elbow' /
    inflection point to prevent the model from being penalized for not
    predicting an unrealistically high RUL early in a unit's life, and to
    avoid encoding a linear degradation assumption where none exists.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'RUL'.
    max_rul : int, default 125
        Upper bound for RUL, matching config.yaml's preprocessing.max_rul.
        125 is the commonly used elbow point for CMAPSS in the literature.

    Returns
    -------
    pd.DataFrame
        DataFrame with a clipped 'RUL' column.
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df


def drop_constant_cols(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """
    Drop known constant or non-informative columns based on the CMAPSS experiment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing operational settings and sensor features.
    experiment : str
        One of {"FD001", "FD002", "FD003", "FD004"}.

    Returns
    -------
    pd.DataFrame
        DataFrame with predefined constant columns removed.
    """
    df = df.copy()

    operational_cols = [
        c for c in df.columns
        if "operational" in c
    ]

    if experiment == "FD001":
        sensors_to_drop = [
            "sensor_1", "sensor_5", "sensor_6",
            "sensor_10", "sensor_16", "sensor_17",
            "sensor_18", "sensor_19"
        ]
        columns_to_drop = operational_cols + sensors_to_drop

    elif experiment == "FD002":
        sensors_to_drop = [
            "sensor_10", "sensor_15", "sensor_16"
        ]
        columns_to_drop = sensors_to_drop

    elif experiment == "FD003":
        sensors_to_drop = [
            "sensor_1", "sensor_5", "sensor_6",
            "sensor_10", "sensor_13", "sensor_16",
            "sensor_18", "sensor_19"
        ]
        columns_to_drop = operational_cols + sensors_to_drop

    elif experiment == "FD004":
        sensors_to_drop = [
            "sensor_10", "sensor_15", "sensor_16"
        ]
        columns_to_drop = sensors_to_drop

    else:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            "Expected one of: FD001, FD002, FD003, FD004."
        )

    return df.drop(columns=columns_to_drop, errors="ignore")
    
