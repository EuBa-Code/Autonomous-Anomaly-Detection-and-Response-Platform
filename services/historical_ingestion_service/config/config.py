import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 

# directory
# MODEL_REGISTRY_PATH = BASE_DIR / "models" 

RAW_DATA_PATH = Path("data") / "historical_data" 

# Use an absolute path for the processed data to avoid relative path confusion in Spark
PROCESSED_DATA = Path("data") / "historical_processed"
TRAIN_LABELS = PROCESSED_DATA / 'train_labels'  
TRAIN = PROCESSED_DATA / 'train'
TEST = PROCESSED_DATA / 'test'