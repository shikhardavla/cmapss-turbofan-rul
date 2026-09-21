#-------------------------------
# sequential.py
# This module creates sequential data that can directly be applied to NN training
#-------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_cols(df: pd.DataFrame):
    """
    Extracts sensor and operational columns dynamically.
    """
    cols = df.columns
    sensor_cols = [c for c in cols if 'sensor' in c]
    op_cols = [c for c in cols if 'operational' in c]

    return sensor_cols, op_cols


def fit_preprocessing(
    df: pd.DataFrame, 
    experiment: str, 
) -> dict:
    """
    Fits scalers and/or KMeans models based on the experiment type.
    Returns a dictionary of stateful objects that can be saved via joblib.
    """
    df = df.copy()
    sensor_cols, op_cols = get_cols(df)
    artifacts = {"experiment": experiment}
    
    if experiment in ["FD001", "FD003"]:
        # Single operating condition: Global scaling
        scaler = StandardScaler()
        scaler.fit(df[sensor_cols])
        
        artifacts["type"] = "global"
        artifacts["sensor_scaler"] = scaler
        
    elif experiment in ["FD002", "FD004"]:
        
        # 1. Fit scaler for operating settings
        settings_scaler = StandardScaler()
        scaled_settings = settings_scaler.fit_transform(df[op_cols])
        
        # 2. Fit KMeans on scaled settings
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(scaled_settings)
        
        # Temporary DataFrame to split sensors by cluster
        temp_df = df[sensor_cols].copy()
        temp_df["cluster_ID"] = cluster_ids
        
        # 3. Fit a separate scaler for the sensors in each cluster
        cluster_scalers = {}
        for c in range(6):
            c_scaler = StandardScaler()
            cluster_data = temp_df[temp_df["cluster_ID"] == c][sensor_cols]
            
            if not cluster_data.empty:
                c_scaler.fit(cluster_data)
                
            cluster_scalers[c] = c_scaler
            
        artifacts["type"] = "regime"
        artifacts["settings_scaler"] = settings_scaler
        artifacts["kmeans"] = kmeans
        artifacts["cluster_scalers"] = cluster_scalers
        
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
        
    return artifacts


def transform_data(
    df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    """
    Applies the fitted artifacts to scale the dataset.
    """
    df = df.copy()
    sensor_cols, op_cols = get_cols(df)

    df[sensor_cols] = df[sensor_cols].astype(float)
    
    if artifacts["type"] == "global":
        scaler = artifacts["sensor_scaler"]
        df[sensor_cols] = scaler.transform(df[sensor_cols])
        
    elif artifacts["type"] == "regime":
        
        # 1. Scale settings using the fitted settings scaler
        settings_scaler = artifacts["settings_scaler"]
        df[op_cols] = settings_scaler.transform(df[op_cols])
        
        # 2. Predict clusters using the fitted KMeans
        kmeans = artifacts["kmeans"]
        df["cluster_ID"] = kmeans.predict(df[op_cols])
        
        # 3. Apply regime-specific sensor scaling
        cluster_scalers = artifacts["cluster_scalers"]
        
        for c, c_scaler in cluster_scalers.items():
            mask = df["cluster_ID"] == c
            if mask.any():
                df.loc[mask, sensor_cols] = c_scaler.transform(
                    df.loc[mask, sensor_cols]
                )
                
    return df


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
):
    """
    Builds 3D tensors for Deep Learning: (Samples, Window Size, Features)
    """
    X, y = [], []
    
    # Ensure strict chronological order
    df = df.sort_values(["unit_number", "time_cycles"])
    
    # Grouping is vastly faster than looping through unique unit numbers
    for unit, unit_df in df.groupby("unit_number"):
        
        # Extract underlying numpy arrays for speed
        data = unit_df[feature_cols].values
        labels = unit_df["RUL"].values
        
        # Corrected sliding window indexing
        # target `labels[i]` is now exactly aligned with the last cycle of the window
        for i in range(window_size - 1, len(unit_df)):
            
            window = data[i - window_size + 1 : i + 1]
            
            X.append(window)
            y.append(labels[i])
            
    return np.array(X), np.array(y)