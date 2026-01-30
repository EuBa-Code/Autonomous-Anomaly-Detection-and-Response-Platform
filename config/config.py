import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if needed
load_dotenv()

class Config:
    # 1. Project Root Definition
    # Since this file is inside 'config/', we go up 2 levels to reach the root.
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"

    # 2. Specific Paths for Synthetic Data Creation
    SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic_data_creation"
    
    # 3. Input and Output Files
    INPUT_DATA_PATH = SYNTHETIC_DIR / "test_data.csv"
    
    # Output: The Parquet file to be generated
    SYNTHETIC_OUTPUT_PATH = SYNTHETIC_DIR / "synthetic_data_sdv.parquet"

    STREAMING_DIR = DATA_DIR / "streaming_data"
    HISTORICAL_DIR = DATA_DIR / "historical_data"
    
    # Column Settings
    TARGET = "Is_Anomaly"
    DROP_COLUMNS = ['Is_Anomaly', 'Anomaly_Type', 'timestamp', 'Machine_ID']