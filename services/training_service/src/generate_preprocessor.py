# generate_preprocessor.py
import logging
import joblib
import pandas as pd
from pathlib import Path

# Import your classes from your project files
# Make sure the import paths match your folder structure
from dataloader import DataLoader
from services.historical_ingestion_service.src.hist_feature_engineering import DataPreprocessor

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArtifactGenerator")

# Configure your paths here
RAW_DATA_PATH = Path("data/historical")  # Where your parquet/csv files are
OUTPUT_PATH = Path("models/preprocessor.joblib")  # Where streaming will look for the file

def generate():
    # 1. Load raw data
    logger.info("Loading data...")
    # DataLoader handles reading CSV/Parquet files
    loader = DataLoader(data_dir=RAW_DATA_PATH)
    try:
        # Try loading parquet files first
        df = loader.load_data(filename="*.parquet")
    except:
        # Fallback to CSV if parquet fails
        df = loader.load_data(filename="*.csv")

    # 2. Initialize the Preprocessor
    # Define columns to EXCLUDE from scaling (ID, timestamp, label)
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
    # We use the modified save_scaler method from before
    preprocessor.save_scaler(OUTPUT_PATH)
    
    logger.info("✅ Done! You can now start streaming.")

if __name__ == "__main__":
    generate()
