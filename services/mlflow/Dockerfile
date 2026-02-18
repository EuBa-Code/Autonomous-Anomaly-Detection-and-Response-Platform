FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --group mlflow
RUN mkdir -p /mlruns/artifacts

EXPOSE 5000

ENV MLFLOW_BACKEND_STORE_URI=file:///mlruns \
    MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns/artifacts \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000 || exit 1

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

