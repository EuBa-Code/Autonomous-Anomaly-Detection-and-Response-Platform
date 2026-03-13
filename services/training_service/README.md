# Training Service

## Overview

**First-run only** training pipeline that fits the initial IsolationForest model from the processed datalake and registers it in MLflow. This service runs **once** to bootstrap the model registry before the online stack starts. All subsequent periodic retraining is handled by `retraining_service`, which uses Feast for feature loading instead.

## File Structure

```
services/training_service/
├── dockerfile
├── config/
│   ├── __init__.py             # Exports Settings instance
│   └── settings.py             # Pydantic settings (MLflow, datalake path, hyperparameters)
└── src/
    ├── __init__.py
    ├── train.py                # Main orchestration entry point
    ├── load_from_datalake.py   # Parquet loader (single file or directory)
    ├── model.py                # sklearn Pipeline factory (preprocessing + IsolationForest)
    ├── evaluator.py            # Metrics calculation and threshold extraction
    └── utils.py                # MLflow model signature helper
```

## Training vs Retraining — Key Differences

| Aspect | `training_service` | `retraining_service` |
|---|---|---|
| Purpose | Bootstrap — run once | Periodic update — every Monday |
| Data source | Datalake (processed Parquet) | Feast point-in-time join |
| Feast dependency | None | Required |
| MLflow experiment | `isolation_forest_prod` | `isolation_forest_retrain` |
| Trigger | Manual / one-shot | Airflow `weekly_retraining` DAG |
| Entity columns dropped | `Machine_ID`, `timestamp` | `Machine_ID`, `created_timestamp` |

## Training Pipeline (`train.py`)

```
1. Load settings (Pydantic + env vars / .env)
        ↓
2. DataManager.load_data()
   processed Parquet → pd.DataFrame
   (single file or partitioned directory)
        ↓
3. Sort chronologically by timestamp, then drop:
   Machine_ID, timestamp (not model features)
        ↓
4. ModelFactory.build_pipeline()
   ColumnTransformer (num: impute+scale | cat: impute+OHE)
   + IsolationForest
        ↓
5. Fit on subsample (≤ 50k rows, RAM cap)
        ↓
6. Chunked inference on full dataset (10k rows/chunk)
   → predictions [-1/+1]  + scores (continuous)
        ↓
7. ProductionMetricsCalculator
   → metrics (anomaly rate, score distribution, latency)
   → thresholds (p01, p05, p50, observed_max_anomaly)
        ↓
8. MLflow run: log params + metrics + model + artifacts
   → version registered under 'if_anomaly_detector'
        ↓
9. thresholds.json written to /outputs (shared Docker volume)
```

## Data Source

Reads from:
```
/data/processed_datasets/machines_with_anomalies_features/
```

This directory is produced by `data_engineering_service` and contains the full feature-enriched dataset including both normal and anomalous records. `DataManager` handles both single `.parquet` files and partitioned directories transparently.

## sklearn Pipeline Architecture

Identical to `retraining_service`:

```
ColumnTransformer
  ├── numerical columns → SimpleImputer(median) → StandardScaler
  └── Cycle_Phase_ID   → SimpleImputer(constant="missing") → OneHotEncoder
          ↓
IsolationForest(n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1)
```

## MLflow Artifacts per Run

| Type | Content |
|---|---|
| Parameters | contamination, max_fit_rows, chunk_size, dataset_size, fit_rows, score_distribution |
| Metrics | anomaly rate, score stats (mean/std/min/max), inference latency |
| Model | full sklearn Pipeline + input/output signature (from raw DataFrame) |
| `thresholds.json` | p01, p05, p50, observed_max_anomaly |
| `metrics.json` | full evaluation snapshot |

The model signature is inferred from the **raw (untransformed) DataFrame** so the inference service can send JSON directly without any pre-processing step.

## Configuration (`settings.py`)

| Setting | Default | Description |
|---|---|---|
| `mlflow_tracking_uri` | `http://mlflow:5000` | MLflow server |
| `mlflow_experiment_name` | `isolation_forest_prod` | MLflow experiment name |
| `mlflow_model_name` | `if_anomaly_detector` | Registered model name |
| `entity_df_path` | `/data/processed_datasets/machines_with_anomalies_features` | Input Parquet path |
| `event_timestamp_column` | `timestamp` | Column used for chronological sort before drop |
| `output_dir` | `/outputs` | Local output directory (shared volume) |
| `max_fit_rows` | `50_000` | Subsampling cap for IsolationForest fit |
| `inference_chunk_size` | `10_000` | Rows per inference chunk during evaluation |
| `training.contamination` | `0.02` | Expected anomaly rate (2%) |
| `training.if_n_estimators` | `100` | Number of trees |
| `training.random_state` | `42` | Reproducibility seed |

## Build & Run

```bash
# Build
docker build -f services/training_service/dockerfile -t training_service:latest .

# Run (once, before starting the online stack)
docker compose run --rm training_service
```

> The inference service loads the model via `models:/if_anomaly_detector/latest`. This service must complete successfully and register a version before `inference_service` can start.