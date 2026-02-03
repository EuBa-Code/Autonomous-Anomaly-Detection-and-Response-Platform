import os
from pathlib import Path

# Get the base directory (one level up from /src)
BASE_DIR = Path(__file__).resolve().parent.parent 

# Use absolute-style paths based on the container root
RAW_DATA_PATH = BASE_DIR / "data" / "historical_data" 
OUTPUT_PATH = BASE_DIR / "models" / "preprocessor.joblib"