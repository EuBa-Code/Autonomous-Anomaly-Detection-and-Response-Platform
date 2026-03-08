"""
config/settings.py
==================
Central configuration for the weekly retraining pipeline.
Uses Pydantic for type validation and .env file support.

All values can be overridden via environment variables or a .env file.

Usage:
    from config.settings import Settings
    s = Settings()
"""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseModel):
    """IsolationForest hyperparameters."""

    contamination: float = Field(default=0.02, description="Expected anomaly rate (e.g. 0.02 = 2%)")
    if_n_estimators: int = Field(default=100,  description="Number of trees in the IsolationForest ensemble")
    random_state: int    = Field(default=42,   description="Seed for reproducibility")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "isolation_forest_retrain"  # separate experiment from first training
    mlflow_model_name: str = "if_anomaly_detector"       # same registered model name → new version each week

    # ── Feast / Feature Store ─────────────────────────────────────────────────
    feast_repo_path: str = "/feature_repo"           # path to feature_store.yaml inside the container
    feature_service_name: str = "machine_anomaly_service_v1"             # Feast FeatureService to use
    entity_df_path: str = "/datalake/telemetry_data/0"  # partitioned parquet directory
    event_timestamp_column: str = "event_timestamp"         # timestamp column name in entity_df

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "/outputs"   # matches Docker volume mount ./outputs:/outputs

    # ── Data processing ───────────────────────────────────────────────────────
    max_fit_rows: int          = 50_000   # subsampling cap for IsolationForest fit (RAM-aware)
    inference_chunk_size: int  = 10_000  # rows per inference chunk during evaluation

    # ── Model hyperparameters ─────────────────────────────────────────────────
    training: TrainingConfig = TrainingConfig()