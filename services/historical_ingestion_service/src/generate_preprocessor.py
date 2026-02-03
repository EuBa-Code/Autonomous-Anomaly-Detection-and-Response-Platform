# generate_preprocessor.py
import logging
import joblib
import pandas as pd
from pathlib import Path
import sys
from config.config import RAW_DATA_PATH, OUTPUT_PATH 

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import your classes - fixed import paths
from dataloader import DataLoader
from hist_feature_engineering import DataPreprocessor

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArtifactGenerator")

# Use relative paths from the container working directory
def generate():
    # 1. Load data
    logger.info("Loading data...")
    
    # ---- Loading the Historical data
    loader = DataLoader(data_dir=RAW_DATA_PATH)
    try:
        # Try loading parquet files first
        logger.info("Attempting to load parquet files...")
        df = loader.load_data(filename="*.parquet")
    except Exception as e:
        logger.warning(f"Parquet loading failed: {e}. Trying CSV...")
        # Fallback to CSV if parquet fails
        try:
            df = loader.load_data(filename="*.csv")
        except Exception as e2:
            logger.error(f"Failed to load data: {e2}")
            raise

    logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")

    # 2. Initialize the Preprocessor
    # Define columns to EXCLUDE from scaling (ID, timestamp, label)
    # ---- Preprocessing data
    preprocessor = DataPreprocessor(
        label_columns=['Is_Anomaly', 'Anomaly_Type', 'Machine_ID', 'timestamp'],
        scaler_type='standard'
    )

    # 3. Fit (compute mean/variance)
    logger.info("Fitting the preprocessor...")
    # This computes statistics and populates self.feature_columns
    preprocessor.preprocess_data(df, fit=True)

    # 4. Save preprocessor
    logger.info(f"Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save_scaler(OUTPUT_PATH)
    
    logger.info("✅ Done! Preprocessor saved successfully.")

if __name__ == "__main__":
    try:
        generate()
    except Exception as e:
        logger.error(f"Error in preprocessor generation: {e}", exc_info=True)
        sys.exit(1)