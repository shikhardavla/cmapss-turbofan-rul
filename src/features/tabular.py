#-------------------------------
# tabular.py
# This module creates tabular data that can directly be applied to RF training
#-------------------------------
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_sensor_columns(df: pd.DataFrame) -> list[str]:

    return [
        c for c in df.columns
        if c.startswith("sensor_")
    ]


def get_operational_columns(df: pd.DataFrame) -> list[str]:

    return [
        c for c in df.columns
        if c.startswith("operational")
    ]


# ---------------------------------------------------------
# Regime clustering (FD002 / FD004 only)
# ---------------------------------------------------------

def add_operating_cluster(
    df: pd.DataFrame,
    experiment: str,
    k: int = 6,
    scaler=None,
    kmeans=None
):
    """
    Note: k=6 matches sequential.py's fit_preprocessing, reflecting the 6
    known operating regimes in FD002/FD004. Keep these in sync — a mismatch
    here would mean the RF and NN pipelines learn a different regime
    segmentation of the same underlying data.

    Sensor values are intentionally left unscaled here (unlike
    sequential.py's per-regime StandardScaler): tree-based models split on
    raw thresholds and are indifferent to feature scale, so no sensor
    scaler is fit or applied in this module.
    """
    df = df.copy()
    if experiment not in ["FD002", "FD004"]:
        return df, scaler, kmeans

    op_cols = get_operational_columns(df)

    # If we don't pass a scaler/kmeans, assume we are in "Train" mode and fit them
    if scaler is None or kmeans is None:
        scaler = StandardScaler()
        op_scaled = scaler.fit_transform(df[op_cols])
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(op_scaled)
    else:
        # We are in "Test" mode, only transform!
        op_scaled = scaler.transform(df[op_cols])
        clusters = kmeans.predict(op_scaled)

    df["cluster_ID"] = clusters
    return df, scaler, kmeans


# ---------------------------------------------------------
# Rolling feature builder
# ---------------------------------------------------------

def make_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
) -> pd.DataFrame:
    """
    Builds rolling mean/std/min/max/slope features per unit, using
    min_periods=1. This is an intentional divergence from
    sequential.create_sequences, which requires a full `window_size` of
    history before emitting a sample.

    Practical effect: for a given unit, cycles 2 through (window - 1) are
    kept with rolling stats computed over fewer than `window` points (only
    the single-point n=1 row is dropped, since its std is NaN). RF is
    tolerant of this — tree splits don't assume comparable statistical
    precision across rows the way distance/gradient-based models do — and
    keeping these rows preserves more training data, especially for
    engines with shorter lifespans. The NN path is stricter because it
    cannot make the same allowance.
    """

    df = df.copy()

    df = df.sort_values(
        ["unit_number", "time_cycles"]
    )

    rolling_gb = (
        df
        .groupby("unit_number")[sensor_cols]
        .rolling(
            window=window,
            min_periods=1,
        )
    )

    # -------------------------
    # Basic stats
    # -------------------------

    stats = rolling_gb.agg(
        ["mean", "std", "min", "max"]
    )

    stats.columns = [
        f"{sensor}_{stat}"
        for sensor, stat in stats.columns
    ]

    # -------------------------
    # Slope features
    # -------------------------

    # -------------------------
    # Slope features
    # -------------------------
    def get_slope(y):
        n = len(y)
        if n < 2:
            return 0.0  # Slope is zero if we only have 1 data point
            
        x_dynamic = np.arange(n)
        x_mean_dyn = x_dynamic.mean()
        denominator_dyn = np.sum((x_dynamic - x_mean_dyn) ** 2)
        
        if denominator_dyn == 0:
            return 0.0
            
        return (
            np.sum((x_dynamic - x_mean_dyn) * (y - y.mean()))
            / denominator_dyn
        )

    slope_dict = {}
    for sensor in sensor_cols:
        slope_series = rolling_gb[sensor].apply(get_slope, raw=True)
        slope_dict[f"{sensor}_slope"] = slope_series

    slope_df = pd.DataFrame(slope_dict)
    stats = pd.concat(
        [stats, slope_df],
        axis=1,
    )

    # -------------------------
    # Restore indices
    # -------------------------

    stats = stats.reset_index(
        level=0,
        drop=True,
    )

    # Keep cluster_ID if present
    join_cols = [
        "unit_number",
        "time_cycles",
        "RUL",
    ]

    if "cluster_ID" in df.columns:

        join_cols.append(
            "cluster_ID"
        )

    stats = df[join_cols].join(
        stats
    )

    stats = stats.dropna()

    stats = stats.reset_index(
        drop=True
    )

    return stats


# ---------------------------------------------------------
# Full pipeline wrapper
# ---------------------------------------------------------

def build_tabular_dataset(
    df: pd.DataFrame,
    window: int,
    experiment: str,
    scaler=None,
    kmeans=None
):
    df = df.copy()

    # --------------------------------
    # Add regime cluster if needed
    # --------------------------------
    # Unpack the tuple correctly
    df, out_scaler, out_kmeans = add_operating_cluster(
        df,
        experiment=experiment,
        scaler=scaler,
        kmeans=kmeans
    )

    sensor_cols = get_sensor_columns(df)

    features_df = make_rolling_features(
        df=df,
        sensor_cols=sensor_cols,
        window=window,
    )

    X = features_df.drop(
        columns=["unit_number", "time_cycles", "RUL"],
        errors="ignore",
    )
    y = features_df["RUL"]

    # Return the fitted objects so train.py can save them!
    return X, y, out_scaler, out_kmeans