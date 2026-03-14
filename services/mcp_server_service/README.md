# MCP Server Service

## Overview

**FastMCP server** that exposes a single RAG tool (`retrieve_context`) consumed by the LangChain agent when investigating an anomaly. It combines hybrid Qdrant retrieval with cross-encoder reranking to return the most relevant knowledge-base excerpts, and logs every query to MongoDB for audit purposes.

## File Structure

```
services/mcp_server_service/
├── Dockerfile
├── server.py                       # FastMCP app — tool definition, health endpoint
├── qdrant/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py               # Qdrant URL, collection, embedding model
│   │   └── qdrant_config.yaml      # Qdrant server config (ports, storage, API key)
│   └── src/
│       ├── __init__.py
│       └── retrieve.py             # Retriever pipeline (hybrid + rerank)
└── mongo_logger/
    ├── config/
    │   ├── __init__.py
    │   └── config.py               # MongoDB URI, database, collection
    └── src/
        ├── __init__.py
        └── logs.py                 # MongoLogger — writes query audit entries
```

## MCP Tool

### `retrieve_context(query: str, machine_id: int) → str`

Called by the LangChain agent with the anomaly description as `query` and the affected machine as `machine_id`. Returns formatted document excerpts with source filenames, or an error string if retrieval fails.

```
agent calls retrieve_context("vibration anomaly machine 1", machine_id=1)
        │
        ├─► Qdrant hybrid retrieval (dense + sparse, MMR, k=6, fetch_k=20)
        │         ↓
        │   FlashrankRerank (ms-marco-TinyBERT-L-2-v2, top_n=4)
        │         ↓
        │   format_docs() → "(Source: machine_1.txt)\n<excerpt>\n\n..."
        │
        └─► MongoLogger.log_query(machine_id) → MongoDB logs_agent collection
```

## Retrieval Pipeline (`retrieve.py`)

Three-stage pipeline built on LangChain:

| Stage | Model / Strategy | Config |
|---|---|---|
| Dense retrieval | `BAAI/bge-m3` (HuggingFace) | Semantic similarity |
| Sparse retrieval | `Qdrant/bm25` (FastEmbed) | Keyword matching |
| Reranking | `ms-marco-TinyBERT-L-2-v2` (Flashrank) | Cross-encoder, `top_n=4` |

- `RetrievalMode.HYBRID` — combines dense and sparse scores in Qdrant
- `search_type="mmr"` — Maximal Marginal Relevance reduces redundancy among the 6 retrieved candidates before reranking
- Final output: 4 reranked chunks passed to the agent as formatted text

## Qdrant Configuration (`qdrant_config.yaml`)

| Setting | Value |
|---|---|
| HTTP port | `6333` |
| gRPC port | `6334` |
| Storage path | `/qdrant/storage` |
| gRPC enabled | `true` |
| Telemetry | disabled |
| API key | set via `QDRANT_API_KEY` env var (`.env`) |

> The ingestion service connects via **gRPC** (`6334`); the MCP server retrieves via **HTTP** (`6333`, `prefer_grpc=False`).

## MongoDB Audit Log (`logs.py`)

Every `retrieve_context` call writes one document to the `logs_agent` collection:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "Machine_ID": 1
}
```

Database: `mcp_database` on `mongodb:27017`. Credentials read from environment (`MONGO_USER` / `MONGO_PASSWORD`, defaulting to `admin/admin`).

## Configuration

### `qdrant/config/config.py`

| Setting | Default | Description |
|---|---|---|
| `qdrant_url` | `http://qdrant:6333` | Qdrant HTTP endpoint |
| `qdrant_collection` | `ingestion_rag_service` | Must match ingestion service collection name |
| `qdrant_api_key` | `None` | Set via `QDRANT_API_KEY` in `.env` |
| `embedding_model` | `BAAI/bge-m3` | Must match the model used during ingestion |

### `mongo_logger/config/config.py`

| Setting | Default | Description |
|---|---|---|
| `mongo_uri` | `mongodb://admin:admin@mongodb:27017` | MongoDB connection string |
| `mongo_db` | `mcp_database` | Database name |
| `collection` | `logs_agent` | Audit log collection |

## Endpoints

| Path | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe — returns `{"status": "ok"}` |
| `/mcp` | — | FastMCP SSE transport (consumed by LangChain service) |

Served on port `8020` (host) inside the Docker network at `http://mcp_server:8020`.

## Build & Run

```bash
# Build
docker build -f services/mcp_server_service/Dockerfile -t mcp_server:latest .

# Run
docker compose --profile online up mcp_server
```

> The compose `start_period` for this service is **600 seconds** — the cross-encoder reranker model load at startup is the bottleneck. The service reports healthy only after it finishes loading.