FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install dependencies FIRST (caching layer)
# Copy service-specific pyproject.toml
COPY utils/offline_files_util/pyproject.toml utils/offline_files_util/uv.lock ./

# No lock file yet, so perform sync without frozen
RUN uv sync --frozen 

COPY utils/offline_files_util/create_offline_files.py/ ./
COPY services/inference_service/config/ config/

CMD ["uv", "run", "-m", "create_offline_files"]

