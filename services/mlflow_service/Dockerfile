FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

ENV UV_PYTHON=python3.11
RUN uv sync --no-dev --group mlflow-server

RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
