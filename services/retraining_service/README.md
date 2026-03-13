# Retraining Service

## Overview

Weekly offline retraining pipeline that pulls the latest features from the Feast feature store, fits a new **IsolationForest** sklearn Pipeline, evaluates it, and registers a new model version in MLflow. Triggered every Monday at 02:00 UTC by the Airflow `weekly_retraining` DAG.

## File Structure

```
services/retraining_service/
├── Dockerfile
├── config/
│   ├── __init__.py         # Exports Settings
│   └── settings.py         # Pydantic settings (MLflow, Feast, hyperparameters)
└── src/
    ├── __init__.py          # Exports FeatureLoader, ModelFactory, ProductionMetricsCalculator, create_and_log_signature
    ├── retrain.py           # Main orchestration entry point
    ├── load_features.py     # Feast point-in-time join → training DataFrame
    ├── model.py             # sklearn Pipeline factory (preprocessing + IsolationForest)
    ├── evaluator.py         # Metrics calculation and threshold extraction
    └── utils.py             # MLflow model signature helper
```

## Training Pipeline (`retrain.py`)

```
1. Load settings (Pydantic + env vars)
        ↓
2. FeatureLoader.load()
   entity_df (parquet) → Feast point-in-time join → pd.DataFrame
        ↓
3. Feature preparation
   - Drop entity columns (Machine_ID, created_timestamp)
   - Cast int columns → float64  (NaN-safe for inference)
   - Cast Cycle_Phase_ID → str   (required by OneHotEncoder)
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
   → new version registered under 'if_anomaly_detector'
        ↓
9. thresholds.json written to /outputs (shared Docker volume)
```

## sklearn Pipeline Architecture

```
ColumnTransformer
  ├── numerical columns → SimpleImputer(median) → StandardScaler
  └── Cycle_Phase_ID   → SimpleImputer(constant="missing") → OneHotEncoder
          ↓
IsolationForest(n_estimators, contamination, random_state, n_jobs=-1)
```

`remainder="drop"` — any column not explicitly listed is silently discarded.

## Feature Loading (`load_features.py`)

`FeatureLoader` handles both single `.parquet` files and partitioned directories, drops QuixStreams metadata columns (`_timestamp`, `_key`) if present, renames `timestamp` → `event_timestamp` (Feast requirement), normalises to UTC, then calls `FeatureStore.get_historical_features()` with the configured `FeatureService`. The timestamp is dropped from the result before returning — it was needed only for the join.

## Evaluation (`evaluator.py`)

No ground-truth labels are available (unsupervised). Evaluation is built on score distributions:

| Metric | Description |
|---|---|
| `n_anomalies_detected` | Records classified as `-1` |
| `anomaly_percentage` | % of total records |
| `score_statistics` | mean / std / min / max of `decision_function` output |
| `score_distribution` | Percentiles p1, p5, p50, p95, p99 |
| `inference_latency_ms` | Average ms per record |

**Thresholds** (`p01`, `p05`, `p50`, `observed_max_anomaly`) are saved as `thresholds.json` both in MLflow and on the shared `/outputs` volume so the inference service can read them without querying MLflow.

## MLflow Artifacts per Run

| Type | Content |
|---|---|
| Parameters | contamination, n_estimators, random_state, dataset size, fit rows, feature service |
| Metrics | anomaly rate, score stats, latency, training time |
| Model | full sklearn Pipeline + input/output signature |
| `thresholds.json` | Score decision boundaries |
| `metrics.json` | Full evaluation snapshot for drift reference |

The model signature is inferred from the **raw (untransformed) DataFrame** — column names and types exactly match what the inference service sends as JSON, with no pre-processing required on the caller side.

## Configuration (`settings.py`)

| Setting | Default | Description |
|---|---|---|
| `mlflow_tracking_uri` | `http://mlflow:5000` | MLflow server |
| `mlflow_experiment_name` | `isolation_forest_retrain` | Experiment name |
| `mlflow_model_name` | `if_anomaly_detector` | Registered model name (new version each run) |
| `feast_repo_path` | `/feature_repo` | Path to `feature_store.yaml` inside container |
| `feature_service_name` | `machine_anomaly_service_v1` | Feast FeatureService to query |
| `entity_df_path` | `/datalake/telemetry_data/0` | Entity DataFrame parquet path |
| `event_timestamp_column` | `event_timestamp` | Timestamp column name for Feast join |
| `output_dir` | `/outputs` | Local output directory (shared volume) |
| `max_fit_rows` | `50_000` | Subsampling cap for IsolationForest fit |
| `inference_chunk_size` | `10_000` | Rows per inference chunk during evaluation |
| `training.contamination` | `0.02` | Expected anomaly rate (2%) |
| `training.if_n_estimators` | `100` | Number of trees |
| `training.random_state` | `42` | Reproducibility seed |

## Build & Run

```bash
# Build
docker build -f services/retraining_service/Dockerfile -t retraining_service:latest .

# Run manually
docker compose run --rm retraining_service
```

In production this is triggered by the Airflow `weekly_retraining` DAG every Monday at 02:00 UTC via `DockerOperator`.