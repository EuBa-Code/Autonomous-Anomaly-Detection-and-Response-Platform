FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /mcp_server_service

# 1. System deps FIRST — before uv sync needs them for compilation
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy only dependency files — this layer is cached until they change
COPY pyproject.toml uv.lock ./

# 3. uv sync with cache mount — packages are cached on your host between builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group mcp-server

# 4. App code last — changes here don't invalidate the deps layer
COPY services/mcp_server_service/ ./

CMD ["uv", "run", "-m", "server"]
