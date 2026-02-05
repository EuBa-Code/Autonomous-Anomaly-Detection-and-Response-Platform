# generate_preprocessor.py
import logging
import sys
from pathlib import Path
from pyspark.sql import SparkSession
import os

# Config imports (assuming these exist in your config file)
from config import RAW_DATA_PATH, MODEL_REGISTRY_PATH, PROCESSED_DATA_PATH

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataloader import DataLoader
from hist_feature_engineering import SparkDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArtifactGenerator")

def generate():
    # 1. Initialize Spark Session
    # We configure it to use enough memory for local processing if needed
    spark = SparkSession.builder \
        .appName("FeatureEngineeringBatch") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    logger.info("Spark Session initialized.")

    try:
        # 2. Load Data
        loader = DataLoader(spark, RAW_DATA_PATH)
        
        # Try Parquet, fall back to CSV if needed (logic handled by spark mostly, 
        # but here we keep it explicit based on your previous logic)
        try:
            df = loader.load_data(file_pattern="*.parquet", file_format="parquet")
        except Exception:
            logger.warning("Parquet not found, trying CSV...")
            df = loader.load_data(file_pattern="*.csv", file_format="csv")

        cols_to_exclude = ['timestamp', 'Machine_ID', 'Is_Anomaly', 'Anomaly_Type']
        
        # 3. Preprocess (Scaling)
        preprocessor = SparkDataPreprocessor(
            label_columns=cols_to_exclude,
            scaler_type='standard'
        )

        # This returns the DF with a new column "features" (Vector)
        df_transformed = preprocessor.fit_transform(df)

        # 4. Save the Model (The Scaler/Pipeline)
        # Spark saves models as folders, not single files
        model_output_path = MODEL_REGISTRY_PATH / "spark_preprocessor"
        
        # Ensure the parent 'models' directory exists
        os.makedirs(str(MODEL_REGISTRY_PATH), exist_ok=True)
        
        preprocessor.save_model(model_output_path)
        logger.info(f"✅ Pipeline Model saved to {model_output_path}")

        # 5. Save the Data for Feast
        # Ensure the processed data directory exists
        os.makedirs(str(PROCESSED_DATA_PATH.parent), exist_ok=True)
        
        df_transformed.write \
            .mode("overwrite") \
            .parquet(str(PROCESSED_DATA_PATH))
            
        logger.info(f"✅ Processed data saved to {PROCESSED_DATA_PATH}")

    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    generate()