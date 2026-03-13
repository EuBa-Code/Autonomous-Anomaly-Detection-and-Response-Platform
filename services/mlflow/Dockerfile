FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/mlflow/pyproject.toml services/mlflow/uv.lock* ./
RUN uv sync --frozen

EXPOSE 5000

ENV PYTHONUNBUFFERED=1