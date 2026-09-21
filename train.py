#---------------------------
# train.py 
# This module applies feature engineering and trains 
# all four architectures on all four datasets
#---------------------------

import os
import yaml
import joblib
import pandas as pd

from src.data.loader import load_cmapss 
from src.data.preprocessor import (
    add_rul,
    clip_rul,
    drop_constant_cols
)
from src.features.tabular import (
    build_tabular_dataset
)
from src.features.sequential import (
    get_cols,
    fit_preprocessing,
    transform_data,
    create_sequences
)
from src.models.randomForest import RandomForestModel
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.models.transformer import TransformerModel


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline():
    config = load_config()
    save_dir = config['experiment']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    datasets = config['experiment']['datasets']
    data_dir = config['experiment']['raw_data_dir']
    window = config['features']['window_size']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f" STARTING PIPELINE: {dataset} ")
        print(f"{'='*50}")
        
        # 1. Load Raw Data (Train only)
        train_df, _, _ = load_cmapss(
            dataset_name=dataset,
            dataset_directory=data_dir
        )
        
        # 2. Universal Preprocessing
        df = train_df.copy()
        df = add_rul(df)
        df = clip_rul(df, max_rul=config['preprocessing']['max_rul'])
        df = drop_constant_cols(df, experiment=dataset)
        
        # =====================================================================
        # 3. TABULAR PIPELINE (Random Forest)
        # =====================================================================
        print("\n--> Engineering Tabular Features...")
        X_tab, y_tab, tab_scaler, tab_kmeans = build_tabular_dataset(
            df=df, 
            window=window, 
            experiment=dataset
        )
        
        # Save tabular artifacts
        tab_artifacts = {"scaler": tab_scaler, "kmeans": tab_kmeans}
        joblib.dump(tab_artifacts, os.path.join(save_dir, f"tab_artifacts_{dataset}.joblib"))
        
        print(f"--> Training Random Forest on {dataset}...")
        rf_model = RandomForestModel(**config['models']['random_forest'])
        rf_model.train(X_tab, y_tab)
        
        # Assuming BaseModel implementations have a .save() method. 
        # If not, use joblib.dump(rf_model.model, ...)
        rf_save_path = os.path.join(save_dir, f"rf_{dataset}.joblib")
        rf_model.save(rf_save_path) 
        print(f"    Saved RF model to {rf_save_path}")

        # =====================================================================
        # 4. SEQUENTIAL PIPELINE (LSTM, TCN, Transformer)
        # =====================================================================
        print("\n--> Engineering Sequential Features...")
        
        # Fit, save artifacts, and transform
        seq_artifacts = fit_preprocessing(df, experiment=dataset)
        joblib.dump(seq_artifacts, os.path.join(save_dir, f"seq_artifacts_{dataset}.joblib"))
        
        df_scaled = transform_data(df, seq_artifacts)
        
        # Identify columns
        sensor_cols, op_cols = get_cols(df_scaled)
        feature_cols = sensor_cols + op_cols
        
        # Build 3D Tensors
        X_seq, y_seq = create_sequences(
            df=df_scaled, 
            feature_cols=feature_cols, 
            window_size=window
        )
        
        # --- Train LSTM ---
        print(f"--> Training LSTM on {dataset}...")
        lstm_model = LSTMModel(**config['models']['lstm'])
        lstm_model.train(X_seq, y_seq)
        lstm_save_path = os.path.join(save_dir, f"lstm_{dataset}.pt")
        lstm_model.save(lstm_save_path)
        print(f"    Saved LSTM model to {lstm_save_path}")

        # --- Train TCN ---
        print(f"--> Training TCN on {dataset}...")
        tcn_model = TCNModel(**config['models']['tcn'])
        tcn_model.train(X_seq, y_seq)
        tcn_save_path = os.path.join(save_dir, f"tcn_{dataset}.pt")
        tcn_model.save(tcn_save_path)
        print(f"    Saved TCN model to {tcn_save_path}")

        # --- Train Transformer ---
        print(f"--> Training Transformer on {dataset}...")
        transformer_model = TransformerModel(**config['models']['transformer'])
        transformer_model.train(X_seq, y_seq)
        transformer_save_path = os.path.join(save_dir, f"transformer_{dataset}.pt")
        transformer_model.save(transformer_save_path)
        print(f"    Saved Transformer model to {transformer_save_path}")

    print("\n" + "="*50)
    print(" ALL TRAINING PIPELINES COMPLETED ")
    print("="*50)


if __name__ == "__main__":
    run_pipeline()