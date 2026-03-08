"""
Central configuration hub.
Uses Pydantic for automatic type validation and .env support.

Feast / feature-store settings have been removed — this pipeline
loads training data exclusively from the datalake.

Usage:
    from config.settings import Settings
    s = Settings()
"""

import os
from pydantic_settings import BaseSettings
from pydantic import BaseModel


class Settings(BaseSettings):

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "isolation_forest_prod"
    mlflow_model_name: str = "if_anomaly_detector"

    # ── Datalake ─────────────────────────────────────────────────────────────
    entity_df_path: str = (
        "/data/processed_datasets/machines_with_anomalies_features"
    )
    event_timestamp_column: str = "timestamp"

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "/outputs"   # matches Docker volume mount ./outputs:/outputs

    # ── Data processing ──────────────────────────────────────────────────────
    max_fit_rows: int = 50_000        # subsample cap for IsolationForest fit
    inference_chunk_size: int = 10_000 # rows per inference chunk

    # ── Model hyperparameters ─────────────────────────────────────────────────
    class TrainingConfig(BaseModel):
        contamination: float = 0.02   # expected anomaly rate (e.g. 0.02 = 2 %)
        if_n_estimators: int = 100    # number of trees in IsolationForest
        random_state: int = 42        # reproducibility seed

    training: TrainingConfig = TrainingConfig()

    class Config:
        env_file = ".env"