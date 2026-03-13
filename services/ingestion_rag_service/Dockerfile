FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=300 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3-dev python3-pip build-essential curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /ingestion_rag_service
COPY services/ingestion_rag_service/pyproject.toml services/ingestion_rag_service/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group ingestion-rag \
    --index "pytorch-cu126=https://download.pytorch.org/whl/cu126"

COPY services/ingestion_rag_service/ ./
CMD ["uv", "run", "-m", "src.ingestion"]