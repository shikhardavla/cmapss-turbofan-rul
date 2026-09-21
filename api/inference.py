import os
from functools import lru_cache
from typing import List

import joblib
import pandas as pd
import yaml

from src.data.preprocessor import drop_constant_cols
from src.features.sequential import get_cols, transform_data
from src.models.lstm import LSTMModel
from src.models.tcn import TCNModel
from src.models.transformer import TransformerModel

from api.schemas import SensorReading


_MODEL_CLASSES = {
    "lstm": LSTMModel,
    "tcn": TCNModel,
    "transformer": TransformerModel,
}

_CONFIG_PATH = os.environ.get("CMAPSS_CONFIG_PATH", "config.yaml")


@lru_cache(maxsize=1)
def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


_config = _load_config()
SAVE_DIR = _config["experiment"]["save_dir"]
WINDOW_SIZE = _config["features"]["window_size"]


@lru_cache(maxsize=None)
def _load_seq_artifacts(dataset: str):
    path = os.path.join(SAVE_DIR, f"seq_artifacts_{dataset}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved artifacts found for dataset '{dataset}' at {path}")
    return joblib.load(path)


@lru_cache(maxsize=None)
def _load_model(dataset: str, model_name: str):
    model_cls = _MODEL_CLASSES[model_name]
    path = os.path.join(SAVE_DIR, f"{model_name}_{dataset}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved '{model_name}' model found for dataset '{dataset}' at {path}")
    return model_cls().load(path)


def _cycles_to_dataframe(dataset: str, cycles: List[SensorReading]) -> pd.DataFrame:
    """
    Rebuilds a raw dataframe matching loader.py's column schema, then applies
    the SAME preprocessing used at train time (drop_constant_cols), so
    downstream feature columns line up exactly with what each model expects.
    """
    records = [c.model_dump() for c in cycles]
    df = pd.DataFrame(records)

    # A placeholder unit_number is required only because drop_constant_cols/
    # get_cols select columns by name prefix -- it's never used as a feature.
    df.insert(0, "unit_number", 1)

    df = drop_constant_cols(df, experiment=dataset)
    return df


def predict_rul(dataset: str, model_name: str, cycles: List[SensorReading]) -> float:
    df = _cycles_to_dataframe(dataset, cycles)

    seq_artifacts = _load_seq_artifacts(dataset)
    df_scaled = transform_data(df, seq_artifacts)

    # Match train.py's exact feature set (sensor + operational columns).
    sensor_cols, op_cols = get_cols(df_scaled)
    feature_cols = sensor_cols + op_cols

    window = df_scaled[feature_cols].values[-WINDOW_SIZE:]
    X = window.reshape(1, WINDOW_SIZE, len(feature_cols))

    model = _load_model(dataset, model_name)
    prediction = model.predict(X)

    return float(prediction[0])