import os
import pandas as pd
from feast import FeatureStore
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureLoader")

def load_historical_features():
    """
    Loads historical features from Parquet into the Online Store (Redis).
    This is necessary for the 'cold start' of the real-time inference.
    """
    repo_path = os.getenv("FEAST_REPO_PATH", "/app")
    store = FeatureStore(repo_path=repo_path)
    
    # Path to historical processed data
    # This should match where hist_ingestion writes its output
    data_dir = os.getenv("DATA_DIR", "/app/data")
    data_path = Path(data_dir) / "processed_datasets/washing_features_batch.parquet"
    
    if not data_path.exists():
        logger.error(f"Historical data not found at {data_path}")
        return

    logger.info(f"Loading historical data from {data_path}...")
    # Read the parquet file
    # We might need to filter or sample if it's too big, 
    # but for Redis we usually want the latest state per entity.
    df = pd.read_parquet(data_path)
    
    # In Feast, to populate the Online Store from historical data, 
    # we usually use 'materialize' or 'materialize-incremental'.
    # For a simple loader, we can use the CLI or store.materialize()
    
    from datetime import datetime, timedelta
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now() + timedelta(days=1)
    
    logger.info(f"Materializing features from {start_date} to {end_date}...")
    store.materialize(start_date, end_date)
    logger.info("Online store population completed!")

if __name__ == "__main__":
    load_historical_features()
