# To use NVIDIA GPU from NVIDIA container toolkit
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /ingestion_rag_service

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --group ingestion-rag

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/ingestion_rag_service/src/ ./src/
COPY services/ingestion_rag_service/config/ ./config/

ENTRYPOINT ["uv", "run", "-m", "src.ingestion"]

