import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    # --- 1. Project Root ---
    # Points to: Progetto-Finale-Master-AI-2025-2026/
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    # --- 2. Kafka Settings ---
    # Matches the 'KAFKA_ADVERTISED_LISTENERS' in your compose.yaml
    KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
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

