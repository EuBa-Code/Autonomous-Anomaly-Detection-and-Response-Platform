FROM apache/airflow:2.10.2-python3.12-bookworm-slim

USER root
# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for PySpark/Postgres
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Set up the environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# Ensure the local bin is in PATH for any scripts
ENV PATH="${PATH}:/home/airflow/.local/bin"

# Install Python dependencies using uv
# We use --system because we want to install into the image's python environment
RUN uv pip install --system --no-cache \
    apache-airflow-providers-apache-spark \
    apache-airflow-providers-redis \
    psycopg2-binary \
    feast[redis] \
    pyspark \
    mlflow