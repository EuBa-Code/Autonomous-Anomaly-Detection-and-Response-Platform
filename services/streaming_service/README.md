# Streaming Service

## Overview

Real-time transformation and feature ingestion service. It consumes raw telemetry from Redpanda, persists a ground-truth raw sink for the batch pipeline, computes two sliding-window feature aggregations, and pushes each window's output directly to the Feast feature server — which writes to both Redis (online store) and Parquet backfill (offline store) in a single HTTP call.

## File Structure

```
services/streaming_service/
├── dockerfile
├── config/
│   └── config.py           # All settings via environment variables
└── src/
    └── app.py              # Full pipeline: sink + derived feature + two windows + Feast push
```

> `feature_store.yaml` is copied from `services/feature_store_service/src/` at build time so the streaming service shares the same Feast project definition.

## Pipeline

```
Redpanda  →  telemetry-data topic
        │
        │  timestamp_extractor (event-time, not broker-time)
        │
        ├─► raw sink  →  LocalFileSink  →  /data/entity_df  (Parquet)
        │                (ground-truth for retraining point-in-time joins)
        │
        ├─► compute Current_Imbalance_Ratio  (per record, before windowing)
        │       = (max(L1,L2,L3) − min(L1,L2,L3)) / mean(L1,L2,L3)
        │
        ├─► 10-min sliding window  (grace: 2 min)
        │       agg: Vibration_RollingMax_10min = Max(Vibration_mm_s)
        │            Machine_ID, latest_timestamp = Latest()
        │       → POST /push  →  vibration_push_source  →  Feast (Redis + Parquet)
        │
        └─► 5-min sliding window  (grace: 2 min)
                agg: Current_Imbalance_RollingMean_5min = Mean(Current_Imbalance_Ratio)
                     Current_Imbalance_Ratio = Latest()
                     Machine_ID, latest_timestamp = Latest()
                → POST /push  →  current_push_source  →  Feast (Redis + Parquet)
```

## Key Design Decisions

**Two push sources, no stateful merge** — each window pushes only its own fields to its own dedicated `PushSource`. This eliminates partial/`None` records that would arise from trying to merge two windows of different lengths into a single push.

**Raw sink before transformation** — every message is written to `/data/entity_df` in its original form *before* any feature computation. These files are the authoritative ground-truth used by the retraining service for Feast point-in-time joins.

**Event-time windowing** — `timestamp_extractor` reads the `timestamp` field embedded in the Kafka payload rather than the broker timestamp. This ensures sliding windows are correctly aligned even when messages arrive slightly late.

**`PUSH_TO = online_and_offline`** — each Feast push writes to both Redis (for real-time inference) and the Parquet backfill path (for historical retrieval). Set to `online` in local development to skip the Parquet write.

## Feast Push Payloads

### `vibration_push_source` (10-min window)
```json
{
  "Machine_ID": 1,
  "timestamp": "2024-01-15T10:30:00+00:00",
  "Vibration_RollingMax_10min": 8.73
}
```

### `current_push_source` (5-min window)
```json
{
  "Machine_ID": 1,
  "timestamp": "2024-01-15T10:30:00+00:00",
  "Current_Imbalance_Ratio": 0.031,
  "Current_Imbalance_RollingMean_5min": 0.027
}
```

## Configuration (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `redpanda:9092` | Redpanda broker |
| `TOPIC_TELEMETRY` | `telemetry-data` | Input topic |
| `AUTO_OFFSET_RESET` | `latest` | Start from latest on first run |
| `FEAST_SERVER_URL` | `http://feature_store_service:6566` | Feast HTTP server |
| `PUSH_SOURCE_VIBRATION` | `vibration_push_source` | Push source name for 10-min window |
| `PUSH_SOURCE_CURRENT` | `current_push_source` | Push source name for 5-min window |
| `FEAST_PUSH_TO` | `online_and_offline` | Write target (`online` / `online_and_offline`) |
| `QUIX_STATE_DIR` | `/tmp/quix_state` | RocksDB state directory for QuixStreams |
| `ENTITY_DF_DIR` | `/data/entity_df` | Raw sink output directory |
| `ENTITY_DF_FORMAT` | `parquet` | Raw sink file format |

## Startup Behaviour

`FeastPusher.wait_until_ready()` blocks for up to 120 seconds polling `GET /health` on the Feast server before the pipeline starts. This prevents push failures during the brief window after Feast starts but before it finishes loading the registry.

## Build & Run

```bash
# Build
docker build -f services/streaming_service/dockerfile -t streaming_service:latest .

# Run
docker compose --profile online up streaming_service
```

Depends on `redpanda` (healthy) and `feature_store_service` (healthy) being up before meaningful data flows.