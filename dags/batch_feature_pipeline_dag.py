"""
DAG: daily_batch_feature_pipeline
==================================
Runs ONCE A DAY (at midnight UTC by default).

Cold-start logic
-----------------
  First ever run  →  `setup_task`
      Materializes ALL existing offline-store rows into Redis via
      `store.materialize_incremental()`, then flips the Airflow Variable
      so this path is never taken again.

  Every subsequent run  →  `run_batch_feature_pipeline`
      Spins up the Spark container that does feature engineering,
      writes to the offline store, AND handles its own incremental
      materialization into Redis — no extra step needed.

After either branch the DAG converges on a `join` EmptyOperator so
any future downstream tasks have a single, clean dependency.

Schedule:
  Adjust `schedule` below to match your preferred daily window,
  e.g. "0 2 * * *" for 02:00 UTC.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

import pendulum
import yaml

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from docker.types import Mount

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Top-level constants — keep in sync with compose.yaml
# HOST_PROJECT_DIR: absolute path on the HOST machine (not inside any container)
# ---------------------------------------------------------------------------
HOST_PROJECT_DIR = Path(os.environ["HOST_PROJECT_DIR"])
SPARK_IMAGE      = os.environ.get("SPARK_IMAGE", "batch_feature_pipeline:latest")
DOCKER_NETWORK   = "anomaly-detection-network"

# Stable task-id strings shared between the branch callable and the task
# definitions so a rename never causes a silent mismatch.
_TASK_SETUP    = "setup_task"
_TASK_STANDARD = "run_batch_feature_pipeline"
_VAR_INIT_KEY  = "batch_pipeline_initialized"

# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    feast_repo_path:     str
    feast_feature_views: List[str]


def _load_settings() -> Settings:
    """
    Reads config.yaml.  The path can be overridden via the CONFIG_PATH
    environment variable (useful when Airflow workers mount the project).
    """
    resolved = os.getenv("CONFIG_PATH", "config.yaml")
    if not Path(resolved).exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with open(resolved) as fh:
        cfg = yaml.safe_load(fh)

    return Settings(
        feast_repo_path     = cfg["feast"]["repo_path"],
        feast_feature_views = cfg["feast"]["feature_views"],
    )


# ---------------------------------------------------------------------------
# Callables
# ---------------------------------------------------------------------------
def check_first_run(**_) -> str:
    """
    Branch callable.
    Returns the task_id of whichever branch should run next.
    """
    initialized = Variable.get(_VAR_INIT_KEY, default_var="false")
    logger.info("Pipeline initialized flag: %s", initialized)
    return _TASK_SETUP if initialized == "false" else _TASK_STANDARD


def cold_start_materialization(**_) -> None:
    """
    Cold-start path — runs exactly once.

    Assumes the offline store already contains the initial batch of rows
    (written by whichever bootstrapping script populated your data lake).
    This task's sole job is to push those rows into Redis so the online
    store is ready before the first real-time inference request arrives.

    After a successful run the Airflow Variable is flipped to 'true' so
    this branch is never taken again.
    """
    # Import here so the DAG file can be parsed even if feast is not
    # installed in the Airflow scheduler environment.
    from feast import FeatureStore

    s        = _load_settings()
    end_date = pendulum.now("UTC")

    logger.info("=== Cold-start materialization ===")
    logger.info("Feast repo       : %s", s.feast_repo_path)
    logger.info("Feature views    : %s", s.feast_feature_views)
    logger.info("Materializing up to: %s", end_date.isoformat())

    store = FeatureStore(repo_path=s.feast_repo_path)
    store.materialize_incremental(
        end_date     = end_date,
        feature_views = s.feast_feature_views,
    )

    logger.info("✓ Cold-start complete — features are now in Redis")
    Variable.set(_VAR_INIT_KEY, "true")


# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------
default_args = {
    "owner":            "airflow",
    "retries":          1,
    "retry_delay":      pendulum.duration(minutes=10),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id         = "daily_batch_feature_pipeline",
    description    = "Runs the Spark batch feature pipeline once a day.",
    start_date     = pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule       = "@daily",   # e.g. "0 2 * * *" for 02:00 UTC
    catchup        = False,      # Don't back-fill missed runs
    max_active_runs = 1,         # Never run two instances in parallel
    default_args   = default_args,
    tags           = ["feature-pipeline", "batch", "daily"],
) as dag:

    # ------------------------------------------------------------------
    # 1. Branch — decide which path to take
    # ------------------------------------------------------------------
    branch = BranchPythonOperator(
        task_id         = "check_if_first_time",
        python_callable = check_first_run,
    )

    # ------------------------------------------------------------------
    # 2a. SETUP PATH (first run only)
    #     Materialize existing offline-store rows → Redis, then mark done.
    # ------------------------------------------------------------------
    setup = PythonOperator(
        task_id         = _TASK_SETUP,
        python_callable = cold_start_materialization,
    )

    # ------------------------------------------------------------------
    # 2b. STANDARD PATH (every subsequent run)
    #     The Spark container handles everything:
    #       feature engineering → offline store → incremental materialization
    #     No separate materialization step is needed.
    # ------------------------------------------------------------------
    run_batch_pipeline = DockerOperator(
        task_id    = _TASK_STANDARD,
        image      = SPARK_IMAGE,
        # The container's ENTRYPOINT is `python`; this becomes the argument.
        command    = ["-m", "batch_pipeline_service.src.batch_pipeline"],
        docker_url = "unix://var/run/docker.sock",
        network_mode = DOCKER_NETWORK,
        environment = {
            "APP_HOME":           "/app",
            "PYTHONUNBUFFERED":   "1",
            "PYTHONPATH":         "/app",
            "SPARK_DRIVER_MEMORY": "2g",
            "CONFIG_PATH":        "/app/batch_pipeline_service/src/config.yaml",
        },
        mounts = [
            Mount(
                source = str(HOST_PROJECT_DIR / "data"),
                target = "/app/data",
                type   = "bind",
            ),
            Mount(
                source = str(HOST_PROJECT_DIR / "services/batch_pipeline_service"),
                target = "/app/batch_pipeline_service",
                type   = "bind",
            ),
            Mount(
                source = str(HOST_PROJECT_DIR / "services/data_engineering_service"),
                target = "/app/data_engineering_service",
                type   = "bind",
            ),
            Mount(
                source    = str(HOST_PROJECT_DIR / "services/feature_store_service/src"),
                target    = "/app/feature_store_service",
                type      = "bind",
                read_only = True,
            ),
        ],
        shm_size          = "2g",        # Spark needs extra shared memory
        auto_remove       = "success",
        mount_tmp_dir     = False,
        execution_timeout = pendulum.duration(hours=2),
    )

    # ------------------------------------------------------------------
    # 3. Join — converge both branches so future tasks have one dependency
    #    trigger_rule="none_failed_min_one_success" lets the join fire even
    #    though one branch was intentionally skipped.
    # ------------------------------------------------------------------
    join = EmptyOperator(
        task_id      = "join",
        trigger_rule = "none_failed_min_one_success",
    )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    branch >> [setup, run_batch_pipeline]
    setup            >> join
    run_batch_pipeline >> join