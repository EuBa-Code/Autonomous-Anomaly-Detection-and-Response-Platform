FROM apache/airflow:2.10.2-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ✅ FIX: Switch to airflow user BEFORE pip/uv — the image blocks pip as root intentionally
USER airflow

# pip works fine as airflow user
RUN pip install --upgrade uv

# ✅ Drop --system — airflow user installs to its own site-packages, no root needed
RUN uv pip install \
    "apache-airflow==2.10.2" \
    "apache-airflow-providers-apache-spark" \
    "apache-airflow-providers-redis" \
    "feast[redis]" \
    "mlflow" \
    "psycopg2-binary" \
    "pyspark" \
    "apache-airflow-providers-docker" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.11.txt"