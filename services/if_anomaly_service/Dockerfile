FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install uv for fast dependency management
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /if_anomaly_service

# Install dependencies FIRST (caching layer)
# Copy service-specific pyproject.toml
COPY services/if_anomaly_service/pyproject.toml services/if_anomaly_service/uv.lock ./

# No lock file yet, so perform sync without frozen
RUN uv sync --frozen --no-dev --group if_anomaly

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code using paths relative to build context (project root)
COPY services/if_anomaly_service/src/ ./src/
COPY services/if_anomaly_service/config/ ./config/

# Default Command
ENTRYPOINT ["uv", "run"]
# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

