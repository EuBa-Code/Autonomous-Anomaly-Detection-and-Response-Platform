# qdrant_service

Vector database service powering similarity search and embedding storage for the anomaly detection pipeline.

Built on top of the official [`qdrant/qdrant`](https://hub.docker.com/r/qdrant/qdrant) image with `curl` added for health-check support.

---

## Ports

| Port | Protocol | Description |
|------|----------|-------------|
| `6333` | HTTP | REST API & Web UI |
| `6334` | gRPC | gRPC API |

---

## Configuration

Runtime behaviour is controlled by a YAML file mounted at container start:

```
services/mcp_server_service/qdrant/config/qdrant_config.yaml
→ /qdrant/config/production.yaml  (inside the container)
```

The following environment variable is also set:

| Variable | Value | Effect |
|----------|-------|--------|
| `QDRANT__STORAGE__ON_DISK_PAYLOAD` | `true` | Stores payload on disk instead of RAM, reducing memory pressure |

---

## Persistent Storage

Vector data is persisted via a bind-mount:

```
./qdrant_data  →  /qdrant/storage
```

This directory is created automatically by Docker on first run. Back it up to avoid losing indexed collections.

---

## Resource Limits

| Limit | Value |
|-------|-------|
| Memory cap | 4 GB |
| Memory reservation | 2 GB |

---

## Health Check

The compose health check hits the Qdrant readiness endpoint:

```
GET http://localhost:6333/healthz
```

- Interval: 10 s — Timeout: 5 s — Retries: 10 — Start period: 30 s

Dependent services should use `condition: service_healthy` to avoid starting before Qdrant is ready.

---

## Network

Attached to `anomaly-detection-network` (bridge). Other services on the same network can reach Qdrant at `http://qdrant:6333`.