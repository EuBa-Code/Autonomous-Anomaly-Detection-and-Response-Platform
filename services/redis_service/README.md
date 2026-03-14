# Redis Service

## Overview

Custom Redis image configured as the **Feast online store** — the low-latency key-value backend that serves pre-computed features to the inference pipeline at scoring time. Persistence is intentionally disabled: Redis holds only the current feature values, which are repopulated from the offline Parquet store via `feast materialize` or streaming pushes.

## File Structure

```
services/redis_service/
├── Dockerfile
└── config/
    └── redis.conf
```

## Base Image

```
redis:6.2-alpine
```

## Configuration (`redis.conf`)

| Setting | Value | Reason |
|---|---|---|
| `save ""` | Persistence disabled | Online feature store — data is always re-materialized from offline store on restart |
| `appendonly no` | AOF disabled | No write-ahead log needed for ephemeral feature cache |
| `maxmemory` | `512mb` | Hard cap to protect the host from OOM |
| `maxmemory-policy` | `allkeys-lru` | Evict the least-recently-used keys when the cap is reached — keeps the freshest features in memory |
| `port` | `6379` | Standard Redis port |
| `bind` | `0.0.0.0` | Accessible from all containers on the Docker network |

## Role in the System

```
feast materialize  ──►  Redis  ◄──  streaming push (per window)
                           │
                           ▼
                    inference service
                    GET /get-online-features
```

Features enter Redis from two directions: the Airflow-triggered `feast materialize-incremental` (batch) and the streaming service's `POST /push` calls (real-time). The inference service reads them via the Feast HTTP server at `redis:6379`.

## Build & Run

```bash
# Build
docker build -f services/redis_service/Dockerfile -t redis_online_store:latest .

# Run
docker compose up redis
```

RedisInsight is available at `http://localhost:5540` for inspecting stored feature keys during development.