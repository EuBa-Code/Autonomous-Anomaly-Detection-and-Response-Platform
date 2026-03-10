FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install uv for fast dependency management
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /mcp_server_service
    
# Install dependencies FIRST (caching layer)
# Copy service-specific pyproject.toml
COPY pyproject.toml uv.lock ./

# No lock file yet, so perform sync without frozen
RUN uv sync --frozen --no-dev --group mcp-server

RUN apt-get update \
&& apt-get install -y --no-install-recommends build-essential \
&& rm -rf /var/lib/apt/lists/*


# Copy application code using paths relative to build context (project root)
COPY services/mcp_server_service/ ./ 

# Default Command
CMD ["uv", "run", "-m", "server"]

