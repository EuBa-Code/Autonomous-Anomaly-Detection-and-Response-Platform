# Anomaly Detection — End to End MLOps System
## Overview

End-to-end, production-ready anomaly detection system for industrial washing machines. The architecture is built around a **dual-pipeline feature store** pattern that eliminates training-serving skew: the same features computed offline for training are served online at inference time, through a single versioned contract.

The focus of this project is **architectural correctness and service connectivity**, not model accuracy. All sensor data is synthetic. The model is an unsupervised `IsolationForest`. The goal is to demonstrate how a real-time ML system integrates streaming features, batch features, model registry, online inference, RAG-based investigation, and operator notification into a single coherent production stack.

<p align="center">
  <image src="docs/Architecture.drawio.svg" width="1200"></image>
</p>

---

## Architecture

The system is composed of five interconnected pipelines. Each is independent but shares state through the feature store, message broker, and model registry.

<p align="center">
  <image src="docs/Pipelines.jpg" width="800"></image>
</p>
---

---

## Project Structure

```
.
├── compose.yaml                        # Full service orchestration
├── dags/
│   ├── dag.py                          # Airflow DAGs (daily batch + weekly retrain)
│   └── config.yaml                     # Feast repo path for DAG tasks
├── docs/                               # Architecture diagrams and images
├── rag_files/
│   ├── machine_1.txt                   # Machine knowledge base (Milnor M-Series)
│   ├── machine_2.txt                   # Machine knowledge base (Girbau GENESIS)
│   └── machine_3.txt                   # Machine knowledge base (Milnor M-Series — Critical)
├── local_models/                       # HuggingFace + vLLM model cache (gitignored)
├── data/                               # All runtime data (gitignored — heavy)
│   ├── synthetic_datasets/             # Raw generated sensor data
│   ├── processed_datasets/             # Feature-enriched data (data engineering output)
│   ├── entity_df/                      # Raw telemetry sink (streaming service output)
│   ├── offline/                        # Feast offline store (batch + streaming backfill)
│   └── registry/                       # Feast registry (SQLite)
├── qdrant_data/                        # Qdrant vector storage (gitignored)
├── redpanda_storage/                   # Redpanda message log storage
├── services/
│   ├── dockerfile.spark_services       # Shared Spark image (batch, data_eng, create_datasets)
│   ├── airflow_service/
│   ├── batch_pipeline_service/
│   ├── create_datasets_service/
│   ├── data_engineering_service/
│   ├── feature_store_service/
│   ├── if_anomaly_service/
│   ├── inference_service/
│   ├── ingestion_rag_service/
│   ├── langchain_service/
│   ├── mcp_server_service/
│   ├── producer_service/
│   ├── redis_service/
│   ├── retraining_service/
│   ├── streaming_service/
│   ├── training_service/
│   └── vllm_service/
└── utils/
    ├── cold_start_util/                # First-run Redis materialization
    └── offline_files_util/             # Feast offline store folder bootstrap
```

## Detailed Data Flow

### 1 — Data Preparation (offline, run once)

```
create_datasets_service
  └─ Generates synthetic sensor data for 3 machines (1M rows, 2% anomaly rate)
  └─ Writes to data/synthetic_datasets/
        │
        ▼
data_engineering_service
  └─ Computes streaming features (rolling windows) + batch features (daily agg)
  └─ Writes enriched Parquet to data/processed_datasets/
        │
        ▼
batch_pipeline_service
  └─ Computes Daily_Vibration_PeakMean_Ratio per machine per day
  └─ Writes to data/offline/machines_batch_features/
  └─ Calls feast.materialize_incremental() → Redis
        │
        ▼
training_service
  └─ Reads processed_datasets (machines_with_anomalies_features)
  └─ Fits sklearn Pipeline: ColumnTransformer + IsolationForest
  └─ Registers model under 'if_anomaly_detector' in MLflow
```

### 2 — Streaming Pipeline (real-time)

```
producer_service
  └─ Reads industrial_washer_with_anomalies_streaming (Parquet)
  └─ Publishes rows to Redpanda [telemetry-data]  (3 msg/s — one per machine)
        │
        ▼
streaming_service  (QuixStreams)
  ├─ raw sink → LocalFileSink → data/entity_df/   ← saved BEFORE any transformation
  │              (ground-truth for retraining point-in-time joins)
  │
  ├─ compute Current_Imbalance_Ratio per record
  │
  ├─ 10-min sliding window
  │     Vibration_RollingMax_10min = Max(Vibration_mm_s)
  │     → POST /push → vibration_push_source → Feast → Redis + Parquet backfill
  │
  └─ 5-min sliding window
        Current_Imbalance_RollingMean_5min = Mean(Current_Imbalance_Ratio)
        → POST /push → current_push_source → Feast → Redis + Parquet backfill
```

### 3 — Batch Pipeline (Airflow — daily)

```
Airflow DAG: daily_batch_feature_pipeline  (00:00 UTC)
  └─ DockerOperator → batch_pipeline_service
        └─ Reads data/entity_df/ (raw telemetry)
        └─ Computes Daily_Vibration_PeakMean_Ratio (PySpark, 1-day window)
        └─ Appends to data/offline/machines_batch_features/
        └─ feast.materialize_incremental() → Redis (machine_batch_features view)
```

### 4 — Inference Pipeline (real-time)

```
inference_service  (QuixStreams)
  └─ Consumes Redpanda [telemetry-data]
  └─ For each message:
       1. GET /get-online-features (machine_anomaly_service_v1) from Feast
             ├─ Vibration_RollingMax_10min          (streaming, TTL 15 min)
             ├─ Current_Imbalance_Ratio              (streaming, TTL 8 min)
             ├─ Current_Imbalance_RollingMean_5min   (streaming, TTL 8 min)
             └─ Daily_Vibration_PeakMean_Ratio       (batch, TTL ~3 years*)
       2. Build feature DataFrame (column order from MLflow model signature)
       3. IsolationForest.predict() + decision_function()
       4. Publish result to Redpanda [predictions]
```

> \* TTL intentionally large for debugging cold-start; production value ~7 days.

**Why Feast as the orchestrator?**
The Feature Store is the single point that joins streaming features (pushed in real time by the streaming service) with batch features (materialized daily by Airflow). The inference service does not know or care where each feature came from — it calls one endpoint and receives the full feature vector.

### 5 — Anomaly Investigation (event-driven)

```
if_anomaly_service
  └─ Consumes Redpanda [predictions]
  └─ Filters: is_anomaly == 1 only
  └─ POST /chat/stream → langchain_service
        │
        ▼
langchain_service  (FastAPI + LangGraph ReAct agent)
  └─ Agent calls MCP tool: retrieve_context(query, machine_id)
        │
        ▼
mcp_server_service  (FastMCP)
  ├─ Qdrant hybrid retrieval (BAAI/bge-m3 dense + BM25 sparse, k=6)
  │     → FlashrankRerank (ms-marco-TinyBERT-L-2-v2, top_n=4)
  │     → returns relevant excerpts from rag_files/ knowledge base
  └─ MongoDB: logs query + machine_id for audit
        │
        ▼
vllm_service  (Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)
  └─ Generates investigation summary with retrieved context
        │
        ▼
langchain_service → notifier.py
  └─ POST Slack webhook: machine ID + full investigation summary
```

### 6 — Retraining Pipeline (Airflow — weekly)

```
Airflow DAG: weekly_retraining  (Monday 02:00 UTC)
  └─ DockerOperator → retraining_service
        └─ Loads entity_df (raw telemetry Parquet from data/entity_df/)
        └─ Feast point-in-time join → full historical feature DataFrame
        └─ Fits new IsolationForest Pipeline (subsample ≤ 50k rows)
        └─ Evaluates on full dataset (chunked, 10k rows/chunk)
        └─ Registers new model version in MLflow under 'if_anomaly_detector'
        └─ Writes thresholds.json to /outputs (shared volume)
```

The inference service always loads `models:/if_anomaly_detector/latest` — no config change needed after a retrain.

---

## Feature Vector at Inference Time

All four features are served by `machine_anomaly_service_v1` from a single Feast call:

| Feature | Source | Cadence | Description |
|---|---|---|---|
| `Vibration_RollingMax_10min` | Streaming | Every message | Max vibration over last 10 min — catches shock events |
| `Current_Imbalance_Ratio` | Streaming | Every message | Instantaneous 3-phase electrical imbalance |
| `Current_Imbalance_RollingMean_5min` | Streaming | Every message | Smoothed imbalance over 5 min — early fault warning |
| `Daily_Vibration_PeakMean_Ratio` | Batch | Daily | Peak/mean vibration over a full day — long-term health score |

---

## Cold Start & Utility Services

Two utility containers handle first-run bootstrapping:

**`cold_start_util`** — On the very first start, the batch pipeline has not yet run and Redis is empty. This utility reads the most recent batch features from the processed historical datasets and materializes them directly into Redis, giving the inference service a valid feature vector from the first message.

**`offline_files_util`** — Creates the offline store directory structure expected by Feast (`data/offline/`) before any pipeline writes to it. Prevents `FileNotFoundError` on first `feast apply`.

---

## Retraining vs Training

| Aspect | `training_service` | `retraining_service` |
|---|---|---|
| Purpose | Bootstrap — run once | Periodic update — every Monday |
| Data source | Processed datalake (Parquet) | Feast point-in-time join |
| Feast dependency | None | Required |
| MLflow experiment | `isolation_forest_prod` | `isolation_forest_retrain` |
| Trigger | Manual / one-shot | Airflow `weekly_retraining` DAG |

---

## Startup Order

### One-time setup (before first online run)

```bash
# 1. Create offline store folders
docker compose run --rm create_offline_files

# 2. Generate synthetic data
docker compose run --rm create_datasets

# 3. Feature engineering
docker compose run --rm data_engineering

# 4. Compute and materialize batch features
docker compose run --rm batch_feature_pipeline

# 5. Register Feast feature definitions
docker compose --profile setup run --rm feature_store_apply

# 6. Initial model training
docker compose run --rm training_service

# 7. Build Qdrant knowledge base (requires GPU)
docker compose run --rm ingestion_rag
```

### Online stack

```bash
# Start all online services
docker compose --profile online up
```

### Airflow (scheduled automation)

```bash
docker compose up airflow-webserver airflow-scheduler airflow-worker
```

---

## Infrastructure at a Glance

| Service | Technology | Port |
|---|---|---|
| Message broker | Redpanda (Kafka-compatible) | `19092` (external) |
| Redpanda Console | Web UI | `8080` |
| Online feature store | Redis 6.2 | `6379` |
| Redis Insight | Web UI | `5540` |
| Feature server | Feast `serve` | `8000` → `6566` |
| ML tracking | MLflow | `5000` |
| Vector database | Qdrant | `6333` (HTTP), `6334` (gRPC) |
| Document database | MongoDB 7 | `27017` |
| LLM server | vLLM (Qwen2.5-7B GPTQ-Int4) | `8222` → `8000` |
| MCP server | FastMCP | `8020` |
| LangChain agent | FastAPI | `8010` |
| Airflow | Web UI | `8081` |

---

## Service Documentation

Each service has its own README with full details on file structure, configuration, and design decisions:

| Service | README |
|---|---|
| Airflow Service | [services/airflow_service/README.md](services/airflow_service/README.md) |
| Batch Pipeline Service | [services/batch_pipeline_service/README.md](services/batch_pipeline_service/README.md) |
| Create Datasets Service | [services/create_datasets_service/README.md](services/create_datasets_service/README.md) |
| Data Engineering Service | [services/data_engineering_service/README.md](services/data_engineering_service/README.md) |
| Feature Store Service | [services/feature_store_service/README.md](services/feature_store_service/README.md) |
| If Anomaly Service | [services/if_anomaly_service/README.md](services/if_anomaly_service/README.md) |
| Inference Service | [services/inference_service/README.md](services/inference_service/README.md) |
| Ingestion RAG Service | [services/ingestion_rag_service/README.md](services/ingestion_rag_service/README.md) |
| LangChain Service | [services/langchain_service/README.md](services/langchain_service/README.md) |
| MCP Server Service | [services/mcp_server_service/README.md](services/mcp_server_service/README.md) |
| Producer Service | [services/producer_service/README.md](services/producer_service/README.md) |
| Redis Service | [services/redis_service/README.md](services/redis_service/README.md) |
| Retraining Service | [services/retraining_service/README.md](services/retraining_service/README.md) |
| Streaming Service | [services/streaming_service/README.md](services/streaming_service/README.md) |
| Training Service | [services/training_service/README.md](services/training_service/README.md) |
| vLLM Service | [services/vllm_service/README.md](services/vllm_service/README.md) |

---

## Requirements

- Docker + Docker Compose
- NVIDIA GPU (required by `ingestion_rag_service` and `vllm_service`)
- NVIDIA Container Toolkit (`runtime: nvidia`)
- ~8 GB VRAM minimum (vLLM: ~4 GB, bge-m3: ~2 GB)
- HuggingFace token (set `HUGGING_FACE_HUB_TOKEN` in `.env` for gated models)
- Slack webhook URL (optional — set `SLACK_WEBHOOK_URL` in `.env` to enable operator notifications)