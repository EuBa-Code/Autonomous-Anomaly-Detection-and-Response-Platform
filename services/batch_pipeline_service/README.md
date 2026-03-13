# Batch Pipeline Service

## Overview

PySpark job that runs **once per day** (triggered by Airflow) to compute `Daily_Vibration_PeakMean_Ratio` — the peak-to-mean vibration ratio per machine per calendar day — and push the result from the Feast offline store into Redis (online store - materialization -) in a single execution.

## File Structure

```
services/
├── dockerfile.spark_services       # Shared image for batch, data_engineering, create_datasets
└── batch_pipeline_service/
    └── src/
        ├── batch_pipeline.py       # Pipeline entry point
        └── config.yaml             # Paths, Spark settings, Feast config
```

> `dockerfile.spark_services` is shared with `data_engineering_service` and `create_datasets_service`. All three services are bundled into the same image and invoked independently via `ENTRYPOINT ["uv", "run"]`.

## Base Image & Key Dependencies

| Layer | Detail |
|---|---|
| Base | `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` |
| System | `openjdk-17-jre-headless` — JVM required by PySpark |
| Python group | `spark-services` (resolved from `pyproject.toml` / `uv.lock`) |
| `JAVA_HOME` | `/usr/lib/jvm/java-17-openjdk-amd64` |

## Computed Feature

```
Daily_Vibration_PeakMean_Ratio = max(Vibration_mm_s) / mean(Vibration_mm_s)
                                  grouped by Machine_ID × calendar day
```

| Value | Interpretation |
|---|---|
| > 1.5 | Spiky / impulsive vibration — potential mechanical fault |
| 1.0 – 1.5 | Smooth operation — healthy machine |
| < 1.0 | Not expected (max ≥ mean by definition) |

## Output Schema

| Column | Type | Notes |
|---|---|---|
| `Machine_ID` | Int64 | Entity key |
| `timestamp` | Timestamp (UTC) | `max(timestamp)` of the daily window — used as Feast `event_timestamp` |
| `Daily_Vibration_PeakMean_Ratio` | Float32 | The computed feature |

## Pipeline Steps

```
1. Read telemetry parquet (entitydf_dir)
        ↓
2. Select Machine_ID, timestamp, Vibration_mm_s
        ↓
3. Group by (Machine_ID, 1-day window) → compute ratio + max timestamp
        ↓
4. Coalesce to 1 partition → write parquet to offline_store_dir  [append]
        ↓
5. feast.materialize_incremental(end=now()) → push to Redis
```

## Configuration (`config.yaml`)

```yaml
paths:
  entitydf_dir:       "/app/data/entity_df/telemetry_data/0"
  offline_store_dir:  "/app/data/offline/machines_batch_features"

spark:
  app_name: "batch-feature-pipeline-washing-machines"
  master:   "local[*]"
  configs:
    spark.sql.session.timeZone:   "UTC"
    spark.sql.shuffle.partitions: "16"
    spark.driver.memory:          "4g"
    spark.executor.memory:        "2g"

schema:
  timestamp_column: "timestamp"

processing:
  write_mode: "append"

feast:
  repo_path:     "/app/feature_store_service"
  feature_views: ["machine_batch_features"]
```

The config path can be overridden at runtime with the `CONFIG_PATH` environment variable.

## Build & Run

```bash
# Build (shared with data_engineering and create_datasets)
docker build -f services/dockerfile.spark_services -t batch_feature_pipeline:latest .

# Run standalone
docker run --rm \
  -e CONFIG_PATH=/app/batch_pipeline_service/src/config.yaml \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/services/batch_pipeline_service:/app/batch_pipeline_service \
  -v $(pwd)/services/feature_store_service/src:/app/feature_store_service \
  batch_feature_pipeline:latest \
  -m batch_pipeline_service.src.batch_pipeline
```

In production this is invoked by the `daily_batch_feature_pipeline` Airflow DAG via `DockerOperator`.