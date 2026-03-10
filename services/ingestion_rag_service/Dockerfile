FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install uv for fast dependency management
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /ingestion_rag_service
    
# Install dependencies FIRST (caching layer)
# Copy service-specific pyproject.toml
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --group ingestion-rag

RUN apt-get update \
&& apt-get install -y --no-install-recommends build-essential \
&& rm -rf /var/lib/apt/lists/*

# Copy application code using paths relative to build context (project root)
COPY services/ingestion_rag_service/ ./ 

# Default Command
CMD ["uv", "run", "-m", "src.ingestion"]

