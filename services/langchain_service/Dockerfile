FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /langchain_service

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY services/langchain_service/pyproject.toml services/langchain_service/uv.lock ./
RUN uv sync --frozen

# Application code
COPY services/langchain_service/config/ config/
COPY services/langchain_service/src/    src/

# ── Environment defaults (override in docker-compose) ──────────────────────
ENV MODE=api \
    MCP_CLIENT_PORT=8010 \
    MCP_SERVER_URL=http://mcp_server:8020 \
    VLLM_BASE_URL=http://vllm:8000/v1 \
    REDPANDA_BROKERS=redpanda:9092 \
    PREDICTIONS_TOPIC=predictions \
    PYTHONUNBUFFERED=1

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

CMD ["uv", "run", "main"]