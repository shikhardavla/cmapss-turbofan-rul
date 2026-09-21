# Turbofan Engine RUL Prediction

This project is an end-to-end machine learning system for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using NASA's C-MAPSS turbofan engine degradation simulation datasets. NASA C-MAPSS is a recognized, research grade simulated dataset within aviation research. This project's focus is less on beating an RMSE leaderboard, but more on the surrounding engineering: a robust train/test discipline, a functional deployment layer, and an evaluation system that incorporates the asymmetric cost function associated with predictions on either side of the ground truth. Additionally, this project explores the use of baseline predictions and metrics towards informing a **mock**, engine maintenance policy system, highlighting the immense potential practical utility of an ML system towards aerospace and mechanical applications.

## Table of Contents

* [Live Demo / Quickstart](#quickstart)
* [Introduction — Why Does This Matter?](#introduction--why-does-this-matter)
* [System Architecture](#system-architecture)
* [Models](#models)
* [Results](#results)
* [Interpretation — What Do the Results Mean?](#interpretation--what-do-the-results-mean)
* [API Reference](#api-reference)
* [Limitations & Future Work](#limitations--future-work)
* [Dataset Attribution](#dataset-attribution)
* [License](#license)
* [Contact](#contact)

## Live Demo / Quickstart

### Run Project Locally

Clone the repository:
```bash
git clone YOUR-REPOSITORY-URL
cd <repo-folder-name>
```

Install the required Python dependencies (full pipeline, including PyTorch — see note below):
```bash
pip install -r requirements.txt
```

> **Note on dependencies:** the root `requirements.txt` covers everything needed to run the full pipeline locally (data loading, feature engineering, all four models, the API). `api/requirements.txt` is a separate, smaller list used only for building the Docker image, which installs a CPU-only PyTorch wheel seperately when running the API. 

Download the raw NASA C-MAPSS dataset (not included in this repository — see [Dataset Attribution](#dataset-attribution) for the official citation) and place the four experiment files for each subset under `data/raw/`, matching the filenames `loader.py` expects:

data/raw/
├── train_FD001.txt test_FD001.txt RUL_FD001.txt
├── train_FD002.txt test_FD002.txt RUL_FD002.txt
├── train_FD003.txt test_FD003.txt RUL_FD003.txt
└── train_FD004.txt test_FD004.txt RUL_FD004.txt

> Official NASA hosting for this dataset has moved over the years and availability can vary; if the citation link is unreachable, search "NASA C-MAPSS Turbofan Jet Engine Data Set" — it's also mirrored on Kaggle and other academic dataset repositories under the same file names.

Create the model artifacts:
```bash
python train.py
python evaluate.py
```
Start the FastAPI service:
```bash
python -m uvicorn api.app:app --reload
```
The API will be available at:

http://localhost:8000/

Serve the dashboard locally (opening `index.html` directly via `file://` will not work — the dashboard's data fetches require a real HTTP origin):
```bash
cd dashboard
python -m http.server 5500
```
Then open `http://localhost:5500` and point the "API URL" field at `http://localhost:8000`.

> **Note:** The trained model artifacts and raw C-MAPSS datasets are not included in the repository. The repository contains the source code and dashboard data (`dashboard/data/*.json`) required to reproduce and explore the project.

### Live Dashboard Coming Soon!

## Introduction — Why Does This Matter?

Aircraft turbofan engines operate under demanding conditions and gradually degrade over time. Accurately anticipating this degradation can enable condition-based maintenance (i.e., how do the operational settings and environment affect planed maintenance). Thi helps reduce unexpected failures, unnecessary maintenance, operational downtime, and in rarer cases, mid-operation engine failure, which can be catastrophic. 

One approach is to predict the **Remaining Useful Life (RUL)** of an engine throughout its operation. RUL represents the estimated number of operational cycles remaining before an engine reaches its defined end-of-life condition. More accurate RUL estimates can help inform maintenance planning and provide an indication of an engine's remaining operational life.

This project develops an end-to-end machine learning system for turbofan engine RUL prediction, from raw sensor data and feature engineering through model training, evaluation, API deployment, and an interactive dashboard.

The project uses NASA's **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** simulated turbofan degradation datasets. Each engine trajectory contains **21 sensor measurements and 3 operational settings** recorded over successive operating cycles. The dataset provides four subsets (FD001–FD004), representing different combinations of operating conditions and degradation modes.

This system compares four machine-learning approaches — **Random Forest, Long Short-Term Memory (LSTM), Temporal Convolutional Network (TCN), and Transformer** — and exposes the trained models through a FastAPI prediction service with an interactive web dashboard.

## System Architecture

```text
                    NASA C-MAPSS Dataset
                             │
                             ▼
                    Data Loading / Cleaning
                           Pandas
                             │
                             ▼
                     Feature Engineering
                   Pandas / Scikit-Learn
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
           Tabular Pipeline       Sequential Pipeline
                  │                     │
                  ▼                     ▼
          Random Forest          LSTM / TCN / Transformer
           Scikit-Learn                  PyTorch
                  │                     │
                  └──────────┬──────────┘
                             ▼
                       Model Evaluation
                    RMSE / MAE / R² / NASA Score
                             │
                             ▼
                    Trained Model Artifacts
                        Joblib / PyTorch
                             │
                             ▼
                       FastAPI /predict
                    FastAPI / Pydantic / Docker
                             │
                             ▼
                    Interactive Dashboard
                       HTML / CSS / JavaScript
```

### Repository Structure

```text
<repo-folder-name>/
│
├── api/
│   ├── app.py
│   ├── schemas.py
│   ├── inference.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   │
│   ├── features/
│   │   ├── tabular.py
│   │   └── sequential.py
│   │
│   └── models/
│       ├── base.py
│       ├── randomForest.py
│       ├── lstm.py
│       ├── tcn.py
│       └── transformer.py
│
├── dashboard/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── data/
│       ├── FD001.json
│       ├── FD002.json
│       ├── FD003.json
│       └── FD004.json
│
├── config.yaml
├── train.py
├── evaluate.py
├── export.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Data Pipeline

The data pipeline converts the raw C-MAPSS FD001–FD004 datasets into representations required by the different models.

Random Forest uses **2D tabular data**, including engineered features such as rolling statistics calculated from previous engine cycles.

The neural-network models use **3D sequential tensors**, where each sample contains a fixed-length history of sensor measurements. These sequences allow the LSTM, TCN, and Transformer to model temporal patterns in engine degradation.

```text
Raw C-MAPSS Data
       │
       ▼
   loader.py
       │
       ▼
 preprocessor.py
       │
       ├───────────────┐
       ▼               ▼
 tabular.py      sequential.py
       │               │
       ▼               ▼
Random Forest    LSTM / TCN / Transformer
```

**`loader.py`** loads the raw text files into Pandas DataFrames and assigns the appropriate feature columns.

**`preprocessor.py`** calculates the RUL target for training data, clips RUL values at 125 cycles, and removes constant or low-variance features according to the project configuration.

**`tabular.py`** generates engineered tabular features for the Random Forest model.

**`sequential.py`** constructs fixed-length sliding windows for the sequential neural-network models.

For example:

```text
Tabular:

engine + cycle + engineered features
                │
                ▼
             one row
                │
                ▼
         Random Forest


Sequential:

engine history
[t-n, ..., t-2, t-1, t]
                │
                ▼
        sequence / tensor
                │
                ▼
       LSTM / TCN / Transformer
```

### API Layer

The trained models are exposed through a FastAPI prediction service.

The `/predict` endpoint accepts the dataset identifier and recent engine sensor history as a validated request defined using Pydantic schemas in `schemas.py`.

`inference.py` handles the model inference pipeline, including preprocessing, model loading, and prediction. `app.py` defines the API routes and connects incoming requests to the inference layer.

The API is containerized with Docker to provide a reproducible deployment environment.

### Dashboard

The dashboard is a static HTML/CSS/JavaScript application that communicates directly with the FastAPI service — it has no backend of its own.

It lets a user pick a C-MAPSS dataset and one or more models, then runs evaluation against real test-set engines (bundled as static JSON under `dashboard/data/`), comparing predicted RUL against ground truth for each unit. Results are shown as both a per-model summary (RMSE, MAE, NASA Score, count of RUL overestimates) and a per-unit detail table, with overestimated predictions flagged — since overestimating remaining life is the more operationally risky error direction.

## Models

### Random Forest

Random Forest is an ensemble of decision trees. For regression, each tree produces an RUL estimate and the final prediction is obtained by aggregating the individual tree predictions.

The model operates on engineered tabular features rather than raw sequential windows.

### LSTM

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to model sequential data.

The network maintains a hidden state and cell state that are updated as each timestep is processed, allowing it to learn temporal relationships within an engine's sensor history.

### TCN

A Temporal Convolutional Network (TCN) uses one-dimensional convolutions to model temporal sequences.

Dilated convolutions allow the network to capture information over a larger temporal receptive field without requiring recurrent processing. Causal convolutions ensure that predictions at a given timestep do not use information from future timesteps.

### Transformer

The Transformer uses self-attention to model relationships between different timesteps within an input sequence.

Unlike recurrent models, the Transformer can process sequence elements in parallel while using attention to determine which parts of the input history are most relevant to the prediction.

## Results

Model performance is evaluated using:

Three standardized ML regression metrics:
* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error
* **R²** — Coefficient of Determination

One system specific metric:
* **NASA Score** — An asymmetric scoring function that penalizes late RUL predictions differently from early predictions. The importance of this is highlighted earlier: an early RUL prediction (i.e., declaring an engine has less life in it than it actually does) likely means scheduling or planning maintenance prior to necessary. This is inefficient at worst, and in some cases actually preferrable. Conversely, overestimating an engine's lifespan consistently means a higher risk of accumulated degredation and failure. 

| Dataset | Model           | RMSE ↓ | MAE ↓ | R² ↑  | NASA Score ↓ |
|---------|-----------------|--------:|-------:|------:|-------------:|
| FD001   | Random Forest   | 18.00   | 12.94  | 0.8125 | 746.34       |
| FD001   | LSTM            | 16.23   | 12.43  | 0.8475 | 456.55       |
| FD001   | TCN             | 17.65   | 13.67  | 0.8196 | 598.17       |
| FD001   | Transformer     | 17.14   | 12.48  | 0.8298 | 396.87       |
| FD002   | Random Forest   | 32.06   | 22.93  | 0.6447 | 49,123.33    |
| FD002   | LSTM            | 29.04   | 19.00  | 0.7043 | 36,498.70    |
| FD002   | TCN             | 28.81   | 19.42  | 0.7092 | 15,157.72    |
| FD002   | Transformer     | 26.35   | 17.97  | 0.7566 | 6,679.50     |
| FD003   | Random Forest   | 17.41   | 12.58  | 0.8230 | 857.62       |
| FD003   | LSTM            | 15.31   | 11.30  | 0.8633 | 570.00       |
| FD003   | TCN             | 19.19   | 14.94  | 0.7851 | 1,016.29     |
| FD003   | Transformer     | 15.78   | 12.27  | 0.8547 | 393.89       |
| FD004   | Random Forest   | 31.91   | 23.91  | 0.6574 | 14,267.97    |
| FD004   | LSTM            | 25.76   | 18.51  | 0.7658 | 4,480.52     |
| FD004   | TCN             | 28.38   | 21.09  | 0.7159 | 5,578.76     |
| FD004   | Transformer     | 24.66   | 17.24  | 0.7854 | 3,900.96     |


## API Reference

### `GET /health`

Returns the current health status of the API.

### `POST /predict`

Accepts an engine's most recent sensor cycles and returns an estimated RUL. Scoped to the sequential models (`lstm`, `tcn`, `transformer`) — Random Forest is not served by this endpoint, since it consumes a different (tabular, rolling-window-statistics) feature representation.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "dataset": "FD001",
           "model": "transformer",
           "unit_id": "engine-42",
           "cycles": [ /* exactly window_size (see config.yaml) cycle objects, in ascending time_cycles order — see api/schemas.py for the full Features schema */ ]
         }'
```

Example response:

```json
{
  "dataset": "FD001",
  "model": "transformer",
  "unit_id": "engine-42",
  "predicted_rul": 42.7
}
```

The exact request schema and available fields are defined in `api/schemas.py`.

## Limitations & Future Work

* RUL targets are clipped at 125 cycles, which simplifies the learning problem but limits predictions above this value.
* C-MAPSS is a simulated dataset and does not represent the full complexity of real-world aircraft engine operation. Thus, the deployed API is intended as a portfolio demonstration rather than of a true aviation maintenance system.
* Future work could include uncertainty estimation, additional degradation datasets, model ensembling, and evaluation on real-world data.

## Dataset Attribution

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**.

The dataset was generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) and contains four datasets representing different combinations of operating conditions and fault modes.

**Dataset citation:**

> A. Saxena and K. Goebel, "Turbofan Engine Degradation Simulation Data Set," NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2008.

The NASA Open Data Portal also identifies the related publication as:

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, 2008.

## License

This project is licensed under the MIT License.

## Contact

**[ Name ]**

* GitHub: [your-github-profile]
* LinkedIn: [your-linkedin-profile]
* Portfolio: [your-portfolio-link]