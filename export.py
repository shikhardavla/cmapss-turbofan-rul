"""
A simple export module that converts CMAPSS data into static JSON 
files for the dashboard to call. Run once prior to deploying dashboard.
"""

import json
import os
import yaml
from src.data.loader import load_cmapss

# Columns match original training features and api\schemas payload for consistency
CMAPSS_COLUMNS = (
    ["time_cycles"]
    + [f"operational_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

OUTPUT_DIR = os.path.join("dashboard", "data")

# loads pipeline config.yaml file
def load_pipeline_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# A function that takes in the dataset name, directory and rolling window size
# and exports data as json files for dashboard lookup
def export_dataset (dataset, data_dir, window_size):
    _, test_df, rul = load_cmapss(dataset, data_dir) 
    rul_lookup = {i+1: int(val) for i, val in enumerate(rul['RUL'].values)}

    units = []
    skipped = []

    for unit in sorted(test_df["unit_number"].unique()):
        unit_df = test_df[test_df["unit_number"] == unit].sort_values("time_cycles")
        last_cycles = unit_df.tail(window_size)

        if len(last_cycles) < window_size:
            skipped.append(unit)
            continue

        cycles = json.loads(last_cycles[CMAPSS_COLUMNS].to_json(orient = "records"))
        units.append({
            "unit": int(unit),
            "true_rul": rul_lookup[unit],
            "cycles": cycles

        })

    if skipped:
        print(f" {dataset}: skipped units with < {window_size} cycles: {skipped}")

    return {"dataset": dataset, "window_size": window_size, "units": units}


def main():
    config = load_pipeline_config()
    data_dir = config["experiment"]["raw_data_dir"]
    window_size = config["features"]["window_size"]
    datasets = config["experiment"]["datasets"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset in datasets:
        print(f"Exporting {dataset} now...")
        payload = export_dataset (dataset=dataset, data_dir=data_dir, window_size=window_size)
        out_path = os.path.join (OUTPUT_DIR, f"{dataset}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f"  Wrote {len(payload['units'])} units to {out_path}")

if __name__ == "__main__":
    main()



