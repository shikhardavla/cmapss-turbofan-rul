import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.loader import load_cmapss
from src.data.preprocessor import drop_constant_cols
from src.features.tabular import (
    make_rolling_features,
    get_sensor_columns,
    add_operating_cluster,
)
from src.features.sequential import get_cols, transform_data
from src.models.randomForest import RandomForestModel
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.models.transformer import TransformerModel


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def evaluate_models():
    config = load_config()
    datasets = config['experiment']['datasets']
    data_dir = config['experiment']['raw_data_dir']
    save_dir = config['experiment']['save_dir']
    window = config['features']['window_size']

    results = []
    feature_importances = {}

    for dataset in datasets:
        print(f"\n{'='*40}")
        print(f"Evaluating Pipeline for {dataset}")
        print(f"{'='*40}")

        # 1. Load test set + ground-truth RUL (one RUL value per engine,
        # corresponding to that engine's final/truncated test cycle).
        _, test_df, rul = load_cmapss(dataset, data_dir)

        # Explicit unit_number -> RUL lookup, rather than relying on
        # positional order matching between test_df and rul.txt.
        rul_lookup = rul.copy()
        rul_lookup["unit_number"] = range(1, len(rul_lookup) + 1)
        rul_lookup = rul_lookup.set_index("unit_number")["RUL"]

        # Apply the SAME column-dropping used at train time before deriving
        # sensor columns -- otherwise LSTM/TCN/Transformer's input_dim won't
        # match what the model was trained on, and RF would be missing
        # cluster_ID for FD002/FD004 (see add_operating_cluster below).
        test_df = drop_constant_cols(test_df, experiment=dataset)
        sensor_cols = get_sensor_columns(test_df)

        # =====================================================================
        # 2. Random Forest Evaluation (Tabular)
        # =====================================================================
        print("--> Preparing Test Features for Random Forest...")

        tab_artifacts_path = os.path.join(save_dir, f"tab_artifacts_{dataset}.joblib")
        tab_artifacts = joblib.load(tab_artifacts_path)
        tab_scaler = tab_artifacts["scaler"]
        tab_kmeans = tab_artifacts["kmeans"]

        test_df_rf = test_df.copy()

        # Reproduce the exact same regime-cluster assignment used at train
        # time (no-op for FD001/FD003; adds cluster_ID for FD002/FD004 using
        # the SAVED scaler/kmeans -- never refit on test data).
        test_df_rf, _, _ = add_operating_cluster(
            test_df_rf,
            experiment=dataset,
            scaler=tab_scaler,
            kmeans=tab_kmeans,
        )

        # RUL is unknown per-row for the test set; make_rolling_features only
        # needs a placeholder column to join against. It's dropped before
        # scoring -- true RUL comes from rul_lookup, evaluated only at each
        # unit's final cycle.
        test_df_rf["RUL"] = 0

        features_rf = make_rolling_features(test_df_rf, sensor_cols, window)
        X_final = features_rf.groupby("unit_number").tail(1)

        y_rf = rul_lookup.loc[X_final["unit_number"]].values
        X_rf = X_final.drop(columns=["unit_number", "RUL", "time_cycles"])

        print("--> Running Random Forest Inference...")
        rf_path = os.path.join(save_dir, f"rf_{dataset}.joblib")
        rf_model = RandomForestModel().load(rf_path)
        rf_predictions = rf_model.predict(X_rf)

        # Feature Importances
        importances = rf_model.model.feature_importances_
        feature_importances[dataset] = pd.DataFrame({
            "Feature": X_rf.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # =====================================================================
        # 3. Sequential Evaluation (LSTM, TCN, Transformer)
        # =====================================================================
        print("--> Preparing Test Features for Sequential Models...")

        seq_artifacts_path = os.path.join(save_dir, f"seq_artifacts_{dataset}.joblib")
        seq_artifacts = joblib.load(seq_artifacts_path)

        test_df_seq = test_df.copy()
        # Reuse the scaler(s)/kmeans fit at train time -- do NOT refit on
        # test data, which would leak test-set statistics into the features.
        test_df_seq = transform_data(test_df_seq, seq_artifacts)

        # Match train.py's exact feature set (sensor + operational columns).
        # cluster_ID is intentionally excluded here, same as train.py.
        sensor_cols_seq, op_cols_seq = get_cols(test_df_seq)
        feature_cols = sensor_cols_seq + op_cols_seq

        X_seq_list = []
        included_units = []
        skipped_units = []

        for unit in sorted(test_df_seq["unit_number"].unique()):
            unit_data = test_df_seq[test_df_seq["unit_number"] == unit]
            last_window = unit_data[feature_cols].values[-window:]

            if len(last_window) < window:
                # Not enough history for this unit to fill a full window;
                # skip rather than silently build a shorter/misaligned one.
                skipped_units.append(unit)
                continue

            X_seq_list.append(last_window)
            included_units.append(unit)

        if skipped_units:
            print(f"    WARNING: skipping units with < {window} cycles: {skipped_units}")

        X_seq = np.array(X_seq_list)  # Shape: (num_engines, window, num_features)
        y_seq = rul_lookup.loc[included_units].values

        print("--> Running LSTM Inference...")
        lstm_path = os.path.join(save_dir, f"lstm_{dataset}.pt")
        lstm_model = LSTMModel().load(lstm_path)
        lstm_predictions = lstm_model.predict(X_seq)

        print("--> Running TCN Inference...")
        tcn_path = os.path.join(save_dir, f"tcn_{dataset}.pt")
        tcn_model = TCNModel().load(tcn_path)
        tcn_predictions = tcn_model.predict(X_seq)

        print("--> Running Transformer Inference...")
        transformer_path = os.path.join(save_dir, f"transformer_{dataset}.pt")
        transformer_model = TransformerModel().load(transformer_path)
        transformer_predictions = transformer_model.predict(X_seq)

        # =====================================================================
        # 4. Calculate Metrics
        # =====================================================================
        
        def nasa_score (y_true, preds):

            error = preds - y_true

            score = np.where(
                error < 0,
                np.exp (-error / 13) - 1,
                np.exp(error / 10) - 1

            )

            return np.sum (score)
        
        model_outputs = [
            ("Random Forest", y_rf, rf_predictions),
            ("LSTM", y_seq, lstm_predictions),
            ("TCN", y_seq, tcn_predictions),
            ("Transformer", y_seq, transformer_predictions),
        ]

        for model_name, y_true, preds in model_outputs:

            assert len(y_true) == len(preds), (
                f"{model_name}: y_true has {len(y_true)} samples, "
                f"but predictions has {len(preds)}"
            )
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            nasa = nasa_score (y_true, preds)

            results.append({
                "Dataset": dataset,
                "Model": model_name,
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "R2": round(r2, 4),
                "NASA Score": round(nasa, 4)
            })

            print(f"    [{model_name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f} | NASA Score: {nasa:.4f}")

    # =====================================================================
    # 5. Save Results
    # =====================================================================
    results_df = pd.DataFrame(results)
    results_path = os.path.join(config['experiment']['save_dir'], "evaluation_metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nAll metrics saved to {results_path}")

    # Optional: sanity check the top 5 RF features for the first dataset
    if datasets and datasets[0] in feature_importances:
        first_dataset = datasets[0]
        print(f"\nTop 5 Features for RF on {first_dataset}:")
        print(feature_importances[first_dataset].head(5))


if __name__ == "__main__":
    evaluate_models()
