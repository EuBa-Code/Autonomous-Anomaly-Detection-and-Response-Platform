import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    # --- 1. Project Root ---
    # Points to: Progetto-Finale-Master-AI-2025-2026/
    APP_HOME = os.getenv('APP_HOME', str(Path(__file__).resolve().parent.parent))
    ROOT_DIR = Path(APP_HOME)
    
    # --- 2. Kafka Settings ---
    # Matches the 'KAFKA_ADVERTISED_LISTENERS' in your compose.yaml
    KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'redpanda:9092')
    TOPIC_TELEMETRY = os.getenv('TOPIC_TELEMETRY', 'telemetry-data')
    CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'anomaly-detector-v1')

    # --- 3. Data Directories ---
    DATA_DIR = ROOT_DIR / "data"
    STREAMING_DIR = DATA_DIR / "streaming_data"
    HISTORICAL_DIR = DATA_DIR / "historical_data"
    SYNTHETIC_DIR = DATA_DIR / "synthetic_data_creation"
    
    # Create directories if they don't exist
    for folder in [STREAMING_DIR, HISTORICAL_DIR, SYNTHETIC_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    # --- 4. File Paths ---
    # The source file for your Producer (Simulation)
    STREAMING_DATASET = STREAMING_DIR / os.getenv('STREAMING_DATASET', 'streaming_data.parquet')
    
    # Paths for synthetic data generation
    INPUT_DATA_PATH = SYNTHETIC_DIR / "test_data.csv"
    SYNTHETIC_OUTPUT_PATH = SYNTHETIC_DIR / "synthetic_data.parquet"

    # --- 5. Machine Learning Features ---
    TARGET = os.getenv('TARGET_COLUMN', 'Is_Anomaly')
    
    # List of columns to ignore during AI Inference
    # These are metadata or labels that would cause 'Data Leakage'
    _drop_cols_str = os.getenv('COLUMNS_TO_DROP', 'Anomaly_Type,timestamp,Machine_ID,Is_Anomaly')
    DROP_COLUMNS = [c.strip() for c in _drop_cols_str.split(',')]

class IsolationForestConfig:
    """Specific settings for the Anomaly Detection Model"""

    BASE_DIR = Config.DATA_DIR
    DATA_DIR = BASE_DIR / "historical_data"
    MODEL_DIR = BASE_DIR / "models"
    METRICS_DIR = BASE_DIR / "metrics"
    
    # Files
    TRAIN_PATH = Config.HISTORICAL_DIR / "train_set.parquet"
    TEST_PATH = Config.HISTORICAL_DIR / "test_set.parquet"
    
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

    # Full output paths
    MODEL_PATH = MODEL_DIR / MODEL_FILENAME
    SCALER_PATH = MODEL_DIR / SCALER_FILENAME
    METRICS_PATH = METRICS_DIR / METRICS_FILENAME

    PREPROCESSOR_JOBLIB = MODEL_DIR / 'preprocessor.joblib'