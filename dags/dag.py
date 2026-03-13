"""
DAG: anomaly_detection_pipeline
=================================
Two independent schedules in a single DAG file:

  1. daily_batch_feature_pipeline  — runs every day at midnight UTC
       Spins up the Spark batch container, runs feature engineering,
       writes to the offline store, and materialises into Redis.

  2. weekly_retraining             — runs every Monday at 02:00 UTC
       Spins up the retraining container, loads features from Feast,
       fits a new IsolationForest, and registers a new model version
       in MLflow.

Both tasks use DockerOperator so the container definitions stay in
sync with the compose.yaml — single source of truth for volumes and env.
"""

import os
import pendulum
from pathlib import Path
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# ---------------------------------------------------------------------------
# Configuration — keep in sync with compose.yaml
# HOST_PROJECT_DIR is injected by the Airflow compose service via ${PWD}.
# ---------------------------------------------------------------------------
HOST_PROJECT_DIR  = Path(os.environ["HOST_PROJECT_DIR"])
SPARK_IMAGE       = os.environ.get("SPARK_IMAGE",      "batch_feature_pipeline:latest")
RETRAINING_IMAGE  = os.environ.get("RETRAINING_IMAGE", "retraining_service:latest")
DOCKER_NETWORK    = "anomaly-detection-network"

default_args = {
    "owner":            "airflow",
    "retries":          1,
    "retry_delay":      pendulum.duration(minutes=10),
    "email_on_failure": False,
}

# ── DAG 1: Daily batch feature pipeline ──────────────────────────────────────
with DAG(
    dag_id="daily_batch_feature_pipeline",
    description="Runs the Spark batch feature pipeline once a day.",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["feature-pipeline", "batch", "daily"],
) as dag_batch:

    DockerOperator(
        task_id="run_batch_feature_pipeline",
        image=SPARK_IMAGE,
        command=["-m", "batch_pipeline_service.src.batch_pipeline"],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment={
            "APP_HOME":            "/app",
            "PYTHONUNBUFFERED":    "1",
            "PYTHONPATH":          "/app",
            "SPARK_DRIVER_MEMORY": "2g",
            "CONFIG_PATH":         "/app/batch_pipeline_service/src/config.yaml",
        },
        mounts=[
            Mount(source=str(HOST_PROJECT_DIR / "data"),                                    target="/app/data",                   type="bind"),
            Mount(source=str(HOST_PROJECT_DIR / "services/batch_pipeline_service"),         target="/app/batch_pipeline_service", type="bind"),
            Mount(source=str(HOST_PROJECT_DIR / "services/data_engineering_service"),       target="/app/data_engineering_service",type="bind"),
            Mount(source=str(HOST_PROJECT_DIR / "services/feature_store_service/src"),      target="/app/feature_store_service",  type="bind", read_only=True),
        ],
        shm_size="2g",
        auto_remove="success",
        mount_tmp_dir=False,
        execution_timeout=pendulum.duration(hours=2),
    )


# ── DAG 2: Weekly retraining ──────────────────────────────────────────────────
with DAG(
    dag_id="weekly_retraining",
    description="Retrains the IsolationForest anomaly detector once a week.",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule="0 2 * * 1",   # every Monday at 02:00 UTC
    catchup=False,
    max_active_runs=1,       # never run two retraining jobs in parallel
    default_args=default_args,
    tags=["retraining", "mlflow", "weekly"],
) as dag_retrain:

    DockerOperator(
        task_id="run_retraining",
        image=RETRAINING_IMAGE,
        command=["python", "retrain.py"],   # mirrors Dockerfile CMD
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        # Mirrors compose.yaml retraining_service environment block exactly.
        environment={
            "PYTHONUNBUFFERED":    "1",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "FEAST_REPO_PATH":     "/feature_store",
            "GIT_PYTHON_REFRESH":  "quiet",
        },
        # Mirrors compose.yaml retraining_service volumes block exactly.
        mounts=[
            Mount(source=str(HOST_PROJECT_DIR / "data/registry"),                       target="/data/registry",  type="bind", read_only=True),
            Mount(source=str(HOST_PROJECT_DIR / "data/offline"),                         target="/data/offline",   type="bind", read_only=True),
            Mount(source=str(HOST_PROJECT_DIR / "data/entity_df"),                       target="/datalake",       type="bind", read_only=True),
            Mount(source=str(HOST_PROJECT_DIR / "services/feature_store_service/src"),   target="/feature_store",  type="bind", read_only=True),
            Mount(source=str(HOST_PROJECT_DIR / "outputs"),                              target="/outputs",        type="bind"),
        ],
        auto_remove="success",
        mount_tmp_dir=False,
        # Retraining should complete well within 1 hour — raise if your dataset is large.
        execution_timeout=pendulum.duration(hours=1),
    )