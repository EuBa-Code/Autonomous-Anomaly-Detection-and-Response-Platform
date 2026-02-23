"""
    Central hub of configuration. Uses Pydantic for:
    - Automatic type validation
    - Reading from environment variables (.env)
    - Clear and centralized structure for all settings
    - Easy extensibility for future configurations (e.g., training parameters)
    All services import this Settings class to access configurations.

    Example of usage:
    from config.settings import Settings
    s = Settings() 
"""


import os
from pydantic_settings import BaseSettings
from pydantic import BaseModel

class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://mlflow:5000"     # MLflow host/port. Local for dev, remote for prod
    mlflow_experiment_name: str = "isolation_forest_prod"  # MLflow experiment name to organize runs
    mlflow_model_name: str = "if_anomaly_detector"         # MLflow Model Registry name. Used for versioning and deployment
    entity_df_path: str = "/data/offline/machines_batch_features" # Path to Parquet file with machine IDs (entity_id, timestamp)
    feast_repo_path: str = "/training_service/feature_store_service/src"  # Path to Feast repository (feature store metadata)
    feature_service_name: str = "machine_anomaly_service_v1" # Name of Feast FeatureService (defines which features to load)
    event_timestamp_column: str = "timestamp"              # Name of timestamp column for temporal data ordering
    output_dir: str = "outputs"                            # Directory to save JSON artifacts (history, thresholds)
    
    feast_chunk_size: int = 50_000      # righe per chunk Feast
    max_fit_rows: int = 200_000         # cap per il fit (subsample)
    inference_chunk_size: int = 50_000  # righe per chunk inference

    class TrainingConfig(BaseModel):
        contamination: float = 0.02                        # Expected percentage of anomalies. If real dataset has <10% anomalies → IF might produce FalsePositive. If >10% → IF might miss TruePositive
        if_n_estimators: int = 100                         # Number of trees in Isolation Forest. Higher = potentially more accurate but slower
        random_state: int = 42                             # Random seed for reproducibility

    training: TrainingConfig = TrainingConfig()            # Default instance for TrainingConfig 

    class Config:                                          # Reads variables from .env if not specified above
        env_file = ".env"