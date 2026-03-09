# To use NVIDIA GPU from NVIDIA container toolkit
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

# Install uv for fast dependency management
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /ingestion_rag_service

# Install dependencies FIRST (caching layer)
# Copy service-specific pyproject.toml
COPY pyproject.toml uv.lock ./

# No lock file yet, so perform sync without frozen
RUN uv sync --frozen --no-dev --group ingestion_rag

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code using paths relative to build context (project root)
COPY services/ingestion_rag_service/src/ ./src/
COPY services/ingestion_rag_service/config/ ./config/

# Default Command
ENTRYPOINT ["uv", "run", "-m", "src.ingestion"]
# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

