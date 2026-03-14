FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

COPY utils/cold_start_util/pyproject.toml utils/cold_start_util/uv.lock ./

RUN uv sync --frozen

COPY services/feature_store_service/src /cold_start_util/feature_store_service
COPY utils/cold_start_util/cold_start.py ./

CMD ["uv", "run", "-m", "cold_start"]
