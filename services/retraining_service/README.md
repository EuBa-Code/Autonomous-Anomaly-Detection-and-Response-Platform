# Retraining Service - Weekly Model Updates

## Overview

The **Retraining Service** performs **weekly scheduled retraining** of the Isolation Forest anomaly detection model using fresh features from the **Feast feature store**. Unlike the initial training service (which loads raw parquet files), this service executes point-in-time joins against Redis and the offline store to ensure no data leakage.

**Key Features:**
- **Feast integration**: Point-in-time feature retrieval (avoids future data leakage)
- **Weekly cadence**: Designed to run via Airflow scheduler
- **Model versioning**: Each run creates a new MLflow model version
- **Production-ready**: Trained models automatically available to inference service

## Architecture

### Retraining Pipeline Flow

```
┌─────────────────────────────────┐
│  Entity DataFrame (Parquet)     │
│  Machine_ID + event_timestamp   │
│  /datalake/telemetry_data/      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Feast Point-in-Time Join       │
│  - Online Store (Redis)         │
│  - Offline Store (Parquet)      │
│  FeatureService: machine_...v1  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Full Feature DataFrame         │
│  (Batch + Streaming Features)   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Drop Entity Columns            │
│  (Machine_ID, created_timestamp)│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Train IsolationForest Pipeline │
│  (Same as initial training)     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  MLflow Model Registry          │
│  Model: if_anomaly_detector     │
│  Version: Auto-incremented      │
│  Experiment: isolation_forest_  │
│              retrain            │
└─────────────────────────────────┘
```

### Key Difference: Point-in-Time Join

**Training Service** (initial):
```
Load parquet → Train model
```

**Retraining Service** (weekly):
```
Entity DF + Feast Point-in-Time Join → Features → Train model
                ↓
    Ensures each row gets features valid at event_timestamp
    (prevents data leakage from future feature values)
```

## File Structure

```
retraining_service/
├── src/
│   ├── __init__.py
│   ├── retrain.py               # Main retraining script (weekly cadence)
│   ├── load_features.py         # Feast point-in-time join loader
│   ├── model.py                 # ModelFactory (same as training)
│   └── utils.py                 # MLflow signature helper
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic config (includes Feast paths)
├── dockerfile
└── README.md                    # This file
```

## Components

### 1. `retrain.py` - Weekly Retraining Script

**Purpose**: Orchestrates weekly model retraining using Feast features

**Key Differences from `train.py`:**
- Loads features from Feast (not raw parquet)
- Uses separate MLflow experiment: `isolation_forest_retrain`
- Drops entity columns: `Machine_ID`, `created_timestamp`
- Handles categorical `Cycle_Phase_ID` conversion to string
- Casts integer columns to float64 (for inference robustness)

**Workflow:**
1. Load entity DataFrame (Machine_ID + event_timestamp)
2. Execute Feast point-in-time join → get full feature set
3. Drop non-feature columns (entity IDs, timestamps)
4. Train IsolationForest pipeline (same as initial training)
5. Log to MLflow → create new model version

### 2. `load_features.py` - Feast Feature Loader

**Purpose**: Execute point-in-time join via Feast feature store

**Class: `FeatureLoader`**
- **Method**: `load()` → pd.DataFrame

**Point-in-Time Join Process:**
1. Load entity_df (single file or partitioned directory)
2. Normalize `event_timestamp` to UTC (Feast requirement)
3. Call `FeatureStore.get_historical_features()`
4. Feast retrieves features valid at each row's timestamp
5. Return clean DataFrame ready for training

**Handles:**
- QuixStreams metadata columns (`_timestamp`, `_key`) → dropped
- Timestamp normalization: `timestamp` → `event_timestamp` (UTC)
- Single parquet file or partitioned directories
- Point-in-time correctness (no future data leakage)

### 3. `model.py` - Pipeline Factory

**Identical to training service** - see training service README for details.

### 4. `utils.py` - MLflow Helpers

**Identical to training service** - see training service README for details.

### 5. `settings.py` - Configuration

**Purpose**: Centralized configuration with Feast integration

**Key Settings (Additional vs Training Service):**
- **Feast**:
  - `feast_repo_path`: "/feature_repo" (path to feature_store.yaml)
  - `feature_service_name`: "machine_anomaly_service_v1"
  - `entity_df_path`: "/datalake/telemetry_data/0" (entity DataFrame location)
  - `event_timestamp_column`: "event_timestamp"
- **MLflow**:
  - `mlflow_experiment_name`: "isolation_forest_retrain" (separate from initial training)
  - `mlflow_model_name`: "if_anomaly_detector" (same name → new versions)

**All other settings** (contamination, max_fit_rows, etc.) are identical to training service.

## Installation

### Prerequisites

- Python 3.12+
- Docker (for containerized deployment)
- Access to MLflow tracking server
- Access to Feast feature store (Redis + offline parquet store)
- Entity DataFrame in parquet format

### Setup (Local)

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd services/retraining_service
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Setup (Docker)

```bash
# Build image
docker build -t retraining-service:latest -f services/retraining_service/dockerfile .

# Run container
docker run --rm \
  -v $(pwd)/feature_repo:/feature_repo \
  -v $(pwd)/datalake:/datalake \
  -v $(pwd)/outputs:/outputs \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e FEAST_REPO_PATH=/feature_repo \
  retraining-service:latest
```

## Configuration

### Example `.env` File

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=isolation_forest_retrain
MLFLOW_MODEL_NAME=if_anomaly_detector

# Feast Feature Store
FEAST_REPO_PATH=/feature_repo
FEATURE_SERVICE_NAME=machine_anomaly_service_v1
ENTITY_DF_PATH=/datalake/telemetry_data/0
EVENT_TIMESTAMP_COLUMN=event_timestamp

# Processing Limits
MAX_FIT_ROWS=50000
INFERENCE_CHUNK_SIZE=10000

# Model Hyperparameters
TRAINING__CONTAMINATION=0.02
TRAINING__IF_N_ESTIMATORS=100
TRAINING__RANDOM_STATE=42
```

### Configuration Differences vs Training Service

| Parameter | Training Service | Retraining Service |
|-----------|------------------|-------------------|
| Data Source | Raw parquet files | Feast feature store (point-in-time join) |
| MLflow Experiment | `isolation_forest_prod` | `isolation_forest_retrain` |
| Entity DF Path | Direct feature parquet | Entity DataFrame (ID + timestamp only) |
| Feast Config | ❌ Not used | ✅ Required (`feast_repo_path`, `feature_service_name`) |

## Usage

### Basic Run (Local)

```bash
cd services/retraining_service
uv run -m src.retrain
```

### Run via Airflow (Recommended)

The service is designed to run weekly via Airflow scheduler:

```python
# In your Airflow DAG
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG('weekly_model_retraining', schedule='@weekly') as dag:
    retrain_task = DockerOperator(
        task_id='retrain_isolation_forest',
        image='retraining-service:latest',
        volumes=[
            '/path/to/feature_repo:/feature_repo',
            '/path/to/datalake:/datalake',
            '/path/to/outputs:/outputs'
        ],
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            'FEAST_REPO_PATH': '/feature_repo'
        }
    )
```

### Expected Output

```
[MAIN] Starting weekly retraining run
[FEAST] Loading entity_df from: /datalake/telemetry_data/0
[FEAST] Found 8 parquet file(s) in directory
[FEAST] entity_df loaded — 250,000 rows
[FEAST] Dropped QuixStreams metadata columns: ['_timestamp', '_key']
[FEAST] Renamed 'timestamp' → 'event_timestamp'
[FEAST] Parsing event_timestamp as UTC...
[FEAST] Connecting to feature store at: /feature_repo
[FEAST] Using FeatureService: 'machine_anomaly_service_v1'
[FEAST] Running point-in-time join (this may take a moment)...
[FEAST] Join completed — 250,000 rows, 15 columns
[FEAST] Feature loading completed in 45.23s
[DATA] Dropping non-feature columns: ['Machine_ID', 'created_timestamp']
[DATA] Cast 5 integer column(s) → float64: ['Vibration_RollingMax_10min', ...]
[DATA] Cast Cycle_Phase_ID → str (categorical)
[DATA] Training features (12): ['Vibration_mm_s', 'Temperature_C', ...]
[DATA] Total rows available: 250,000
[PIPELINE] Numeric features: 11 | Categorical features: 1
[TRAIN] Subsampling 250,000 → 50,000 rows (RAM cap)
[TRAIN] Fitting pipeline on 50,000 rows...
[TRAIN] Training completed in 9.87s
[INFERENCE] 250,000 rows → 25 chunk(s) of 10,000
[INFERENCE] Chunk 1/25 — 10,000 rows
...
[INFERENCE] Completed in 18.45s | 0.074 ms/record
[METRICS] Anomalies: 5,000 (2.00%)
[METRICS] Normal: 245,000 (98.00%)
[METRICS] Score mean: -0.0987 | std: 0.0423
[ARTIFACTS] Exported metrics to /outputs/metrics.json
[MLFLOW] New model version registered under 'if_anomaly_detector'
[MAIN] Weekly retraining run completed successfully!
```

## Point-in-Time Join Explained

### Why Point-in-Time Joins Matter

**Problem:** Naive joins can cause **data leakage** (using future information to predict the past)

**Example:**
```
Event at 10:00 AM → Feature computed at 10:30 AM
❌ Naive join: Uses 10:30 AM feature value (leaked future data)
✅ Point-in-time join: Uses feature value from 10:00 AM or earlier
```

**How Feast Solves This:**
1. For each row's `event_timestamp`, Feast retrieves the feature value that was **valid at that exact moment**
2. If a feature was updated at 10:15 AM, a 10:00 AM event gets the **previous** value (from before 10:15)
3. Ensures training data matches production reality (no future features)

### Feast Data Flow

```
Entity DF                    Feast Offline Store
┌──────────────┐            ┌─────────────────────────┐
│ Machine_ID │ TS │         │ Machine_ID │ TS │ Feat1 │
├──────────────┤            ├─────────────────────────┤
│ 1 │ 10:00 AM │  ────────▶│ 1 │ 09:45 │ 2.3 │ ✓ Used
│ 1 │ 10:30 AM │  ────────▶│ 1 │ 10:15 │ 3.1 │ ✓ Used
│ 2 │ 09:50 AM │  ────────▶│ 1 │ 10:45 │ 2.8 │ ✗ Future
└──────────────┘            │ 2 │ 09:30 │ 1.5 │ ✓ Used
                            └─────────────────────────┘
                                     ▼
                            Point-in-Time Join Result
                            ┌──────────────────────────┐
                            │ Machine_ID │ TS │ Feat1 │
                            ├──────────────────────────┤
                            │ 1 │ 10:00 │ 2.3 │ (from 09:45)
                            │ 1 │ 10:30 │ 3.1 │ (from 10:15)
                            │ 2 │ 09:50 │ 1.5 │ (from 09:30)
                            └──────────────────────────┘
```

## Integration with Feature Store

### Required Feast Setup

Before running retraining, ensure:

1. **Feature Store Initialized:**
```bash
cd feature_repo
feast apply
```

2. **Offline Store Populated:**
```bash
# Batch features materialized
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%SZ)
```

3. **Entity DataFrame Available:**
```bash
# Check entity_df exists
ls /datalake/telemetry_data/0/*.parquet
```

### Feature Service Definition

The retraining service uses a **FeatureService** to group features:

```python
# In feature_repo/features.py
from feast import FeatureService

machine_anomaly_service_v1 = FeatureService(
    name="machine_anomaly_service_v1",
    features=[
        machine_streaming_features,  # Real-time sensor features
        machine_batch_features,      # Daily aggregates
    ],
)
```

## MLflow Integration

### Experiment Structure

**Initial Training**:
- Experiment: `isolation_forest_prod`
- Purpose: First model training from raw data
- Runs: Manual/one-time

**Weekly Retraining**:
- Experiment: `isolation_forest_retrain`
- Purpose: Scheduled weekly updates
- Runs: Automated via Airflow

**Shared Model Registry**:
- Model Name: `if_anomaly_detector`
- Versioning: Auto-incremented (v1, v2, v3...)
- Promotion: Manual stage transitions (None → Staging → Production)

### Model Versioning Workflow

```
Week 1: Retrain → v2 (Staging)
Week 2: Retrain → v3 (Staging)
Week 3: v3 promoted to Production
        Retrain → v4 (Staging)
Week 4: v4 promoted to Production
        Retrain → v5 (Staging)
```

## Data Flow

```
┌────────────────────────────────────────────┐
│  Streaming Pipeline                        │
│  - Real-time features → Redis              │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Batch Pipeline                            │
│  - Daily features → Offline Store          │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Feast Feature Store                       │
│  - Online: Redis (streaming features)      │
│  - Offline: Parquet (batch features)       │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Entity DataFrame (Datalake)               │
│  - QuixStreams output: Machine_ID + TS     │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Retraining Service (THIS SERVICE)         │
│  - Point-in-time join via Feast            │
│  - Train new model version                 │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  MLflow Model Registry                     │
│  - New version created weekly              │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│  Inference Service                         │
│  - Auto-reload latest Production model     │
└────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: Feast connection error

**Cause**: Feast repo path incorrect or feature store not initialized

**Solution**:
```bash
# Verify feature_store.yaml exists
ls /feature_repo/feature_store.yaml

# Re-apply Feast definitions
cd /feature_repo
feast apply

# Check registry
feast registry-dump
```

### Issue: Empty DataFrame after Feast join

**Cause**: Entity DF timestamps outside feature store data range

**Solution**:
```bash
# Check entity_df timestamp range
python -c "
import pandas as pd
df = pd.read_parquet('/datalake/telemetry_data/0')
print(df['timestamp'].min(), df['timestamp'].max())
"

# Check offline store materialization range
feast materialize-incremental --help
```

### Issue: KeyError: 'event_timestamp'

**Cause**: Timestamp column not named correctly

**Solution**:
```bash
# Check actual column name in entity_df
python -c "
import pandas as pd
df = pd.read_parquet('/datalake/telemetry_data/0')
print(df.columns.tolist())
"

# Update settings.py or .env
export EVENT_TIMESTAMP_COLUMN=timestamp  # or actual column name
```

### Issue: Slow point-in-time join (>5 minutes)

**Cause**: Large entity_df or unoptimized offline store

**Solution**:
```bash
# Reduce entity_df size (subsample if acceptable)
export MAX_FIT_ROWS=20000

# Check offline store format (parquet preferred over SQL)
# Ensure offline store is partitioned by date

# Optimize Feast config (in feature_store.yaml)
# Use parquet provider instead of SQLite for large datasets
```

## Performance Tuning

### For Weekly Production Runs

```bash
# Recommended configuration
MAX_FIT_ROWS=100000              # Use more data for stable models
INFERENCE_CHUNK_SIZE=20000       # Larger chunks (faster)
TRAINING__IF_N_ESTIMATORS=150    # More trees (better accuracy)
```

### For Development/Testing

```bash
# Fast iteration configuration
MAX_FIT_ROWS=5000                # Quick training
INFERENCE_CHUNK_SIZE=1000        # Minimal RAM
TRAINING__IF_N_ESTIMATORS=50     # Fewer trees
```

## Related Files

- **`training_service/`**: Initial model training (raw parquet input)
- **`feature_repo/`**: Feast feature definitions
- **`batch_pipeline/`**: Daily feature computation
- **`streaming_pipeline/`**: Real-time feature updates
- **`inference_service/`**: Model deployment and serving
- **`airflow/`**: Orchestration DAGs (weekly retraining schedule)

## Next Steps

1. **Verify Feast Setup**: Ensure feature store is initialized and materialized
2. **Configure Paths**: Update `.env` with correct Feast repo and entity_df paths
3. **Test Manually**: Run `uv run -m src.retrain` to verify end-to-end
4. **Schedule in Airflow**: Add weekly DAG for automated retraining
5. **Monitor MLflow**: Check new model versions appear in registry
6. **Promote Models**: Move best versions to "Production" stage

## Key Metrics to Monitor

**Weekly Retraining Health:**
- **anomaly_rate**: Should remain stable (~2%) across weeks
- **score_mean/std**: Large shifts indicate data distribution changes
- **train_time_sec**: Sudden increases suggest data growth
- **feature_loading_time**: Monitor Feast join performance

**Model Quality Drift:**
- Compare score distributions across model versions
- Alert if anomaly_rate deviates >5% from expected contamination
- Track p95/p99 percentiles for threshold stability