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
    # MLflow configuration
    mlflow_tracking_uri: str = "http://mlflow:5000"                    # MLflow server URI
    mlflow_experiment_name: str = "isolation_forest_prod"              # MLflow experiment name
    mlflow_model_name: str = "if_anomaly_detector"                     # Model Registry name
    
    # ============================================================
    # DATA LOADING: Point directly to offline store processed data
    # ============================================================
    # Path to offline store directory with processed features
    # Can be:
    #   - Single parquet file: /data/processed_datasets/data.parquet
    #   - Directory with parquet files: /data/processed_datasets/industrial_washer_normal_features/
    #   - Spark output (multiple part files): /data/processed_datasets/industrial_washer_normal_features/
    offline_store_path: str = "data/processed_datasets/industrial_washer_normal_features"
    
    # Timestamp column name in your offline store data
    event_timestamp_column: str = "timestamp"                          # Change if different in your data
    
    # Feature store repo (kept for reference, not used in simple mode)
    feast_repo_path: str = "/training_service/feature_store_service/src"
    
    # ============================================================
    # DATA PROCESSING
    # ============================================================
    output_dir: str = "outputs"                                         # Directory for JSON artifacts
    max_fit_rows: int = 200_000                                        # Subsample size for training
    inference_chunk_size: int = 50_000                                 # Batch size for inference
    
    # ============================================================
    # MODEL TRAINING
    # ============================================================
    class TrainingConfig(BaseModel):
        contamination: float = 0.02                        # Expected % of anomalies
        if_n_estimators: int = 100                         # Isolation Forest trees
        random_state: int = 42                             # Random seed

    training: TrainingConfig = TrainingConfig()            # Default instance

    class Config:
        env_file = ".env"