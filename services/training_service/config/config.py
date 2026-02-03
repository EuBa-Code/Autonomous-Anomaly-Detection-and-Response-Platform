import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class IsolationForestConfig:
    
    APP_HOME = os.getenv('APP_HOME', str(Path(__file__).resolve().parent.parent))
    ROOT_DIR = Path(APP_HOME)
    DATA_DIR = ROOT_DIR / "data"

    """Specific settings for the Anomaly Detection Model"""

    BASE_DIR = DATA_DIR
    DATA_DIR = BASE_DIR / "data/historical_data"
    MODEL_DIR = BASE_DIR / "models"
    METRICS_DIR = BASE_DIR / "metrics"

    HISTORICAL_DIR = DATA_DIR / "historical_data"
    SYNTHETIC_DIR = DATA_DIR / "synthetic_data_creation"
    
    # Files
    TRAIN_PATH = HISTORICAL_DIR / "train_set.parquet"
    TEST_PATH = HISTORICAL_DIR / "test_set.parquet"
    
    # Isolation Forest parameters
    ISOLATION_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_samples': 'auto',
        'contamination': 0.1,  # Expected percentage of anomalies
        'max_features': 1.0,
        'bootstrap': False,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }

    MODEL_FILENAME = 'isolation_forest_model.pkl'
    METRICS_FILENAME = "training_metrics.json"
    SCALER_FILENAME = "scaler.pkl"

    MODEL_DIR = DATA_DIR / 'models'
    METRICS_DIR = DATA_DIR / 'metrics'

    # Full output paths
    MODEL_PATH = MODEL_DIR / MODEL_FILENAME
    SCALER_PATH = MODEL_DIR / SCALER_FILENAME
    METRICS_PATH = METRICS_DIR / METRICS_FILENAME

    PREPROCESSOR_JOBLIB = MODEL_DIR / 'preprocessor.joblib'