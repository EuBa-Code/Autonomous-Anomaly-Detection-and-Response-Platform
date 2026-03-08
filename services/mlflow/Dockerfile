FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --group mlflow

EXPOSE 5000

ENV PYTHONUNBUFFERED=1