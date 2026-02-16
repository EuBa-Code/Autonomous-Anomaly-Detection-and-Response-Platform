"""
Configuration module for Training Service.
Centralizes all paths and parameters.
"""
import os
from pathlib import Path


class TrainingConfig:
    """
    Configuration class for the Training Service.
    All paths and parameters should be defined here.
    """
    
    # ============================================================================
    # PATHS
    # ============================================================================
    
    # Root directories
    APP_ROOT = os.getenv('APP_ROOT', '/training')
    ROOT_DIR = Path(APP_ROOT)  # Container root
    
    DATA_DIR = Path(os.getenv('DATA_DIR', str(ROOT_DIR / "data")))
    MODELS_DIR = Path(os.getenv('MODELS_DIR', str(ROOT_DIR / "models")))
    METRICS_DIR = Path(os.getenv('METRICS_DIR', str(ROOT_DIR / "metrics")))
    
    # Data subdirectories
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # ============================================================================
    # FILE PATTERNS
    # ============================================================================
    
    TRAIN_FILE_PATTERN = "train*.parquet"
    TEST_FILE_PATTERN = "test*.parquet"
    
    # ============================================================================
    # ARTIFACT NAMES
    # ============================================================================
    
    MODEL_ARTIFACT = "isolation_forest_model.pkl"
    PREPROCESSOR_ARTIFACT = "preprocessor.pkl"
    METRICS_ARTIFACT = "training_metrics.json"
    PREDICTIONS_FILE = "test_predictions.csv"
    
    # ============================================================================
    # MODEL PARAMETERS
    # ============================================================================
    
    ISOLATION_FOREST_PARAMS = {
        'n_estimators': 100,
        'contamination': 0.1,  # Expected proportion of anomalies
        'max_samples': 'auto',
        'random_state': 42,
        'n_jobs': -1,  # Use all available cores
        'verbose': 0
    }
    
    # ============================================================================
    # PREPROCESSING
    # ============================================================================
    
    # Columns to exclude from features (labels, IDs, timestamps)
    LABEL_COLUMNS = ['Is_Anomaly', 'Anomaly_Type', 'Machine_ID', 'timestamp']
    
    # ============================================================================
    # LOGGING
    # ============================================================================
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = ROOT_DIR / "logs" / "training.log"


# Create a singleton instance
config = TrainingConfig()