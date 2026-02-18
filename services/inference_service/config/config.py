import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    # --- 1. Project Root ---
    # Points to /inference_service (in container)
    APP_HOME = os.getenv('APP_HOME', str(Path(__file__).resolve().parent.parent))
    ROOT_DIR = Path(APP_HOME)
    
    # --- 2. Kafka Settings ---
    KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'redpanda:9092')
    TOPIC_TELEMETRY = os.getenv('TOPIC_TELEMETRY', 'telemetry-data')
    CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'inference-service-v1')

    # --- 3. Data Directories ---
    DATA_DIR = ROOT_DIR / "data"
    
    # --- 5. Machine Learning Features ---
    TARGET = os.getenv('TARGET_COLUMN', 'Is_Anomaly')
    
    # List of columns to ignore during AI Inference
    _drop_cols_str = os.getenv('COLUMNS_TO_DROP', 'Anomaly_Type,timestamp,Machine_ID,Is_Anomaly')
    DROP_COLUMNS = [c.strip() for c in _drop_cols_str.split(',')]

class IsolationForestConfig:
    """Specific settings for the Anomaly Detection Model"""

    BASE_DIR = Config.DATA_DIR
    # Where models are mounted
    MODEL_DIR = BASE_DIR / "models"
    
    # Files
    MODEL_FILENAME = 'isolation_forest_model.pkl'
    SCALER_FILENAME = "scaler.pkl"
    PREPROCESSOR_JOBLIB = MODEL_DIR / 'preprocessor.joblib'

    # Full output paths
    MODEL_PATH = MODEL_DIR / MODEL_FILENAME
    SCALER_PATH = MODEL_DIR / SCALER_FILENAME
