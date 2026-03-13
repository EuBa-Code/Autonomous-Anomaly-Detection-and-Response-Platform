FROM python:3.11.9-slim

WORKDIR /retraining_service

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Cache dependency install before copying source code
COPY services/retraining_service/pyproject.toml services/retraining_service/uv.lock* ./
RUN uv sync --frozen 

# Copy service source
COPY services/retraining_service /retraining_service

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/retraining_service

CMD ["uv", "run", "python", "-m", "src.retrain"]