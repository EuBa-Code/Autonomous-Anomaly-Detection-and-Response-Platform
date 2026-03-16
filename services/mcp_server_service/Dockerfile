FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /mcp_server_service

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/mcp_server_service/pyproject.toml services/mcp_server_service/uv.lock ./

RUN uv sync --frozen

COPY services/mcp_server_service/ ./

CMD ["uv", "run", "-m", "server"]
