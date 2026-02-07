# generate_preprocessor.py
import logging
import sys
from pathlib import Path
from pyspark.sql import SparkSession
import os

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

sys.path.insert(0, str(Path(__file__).parent))

from dataloader import DataLoader
from hist_feature_engineering import SparkDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureEngineering")

def generate():
    # 1. Initialize Spark Session with timestamp handling configs
    spark = SparkSession.builder \
        .appName("FeatureEngineeringBatch") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.nanosecondsAsLong", "true") \
        .getOrCreate()
    
    logger.info("Spark Session initialized.")

    try:
        # 2. Load Data
        loader = DataLoader(spark, RAW_DATA_PATH)
        
        # Try Parquet, fall back to CSV if needed
        try:
            logger.info("Attempting to load Parquet files...")
            df = loader.load_data(file_pattern="train_set.parquet", file_format="parquet")
        except Exception as parquet_error:
            logger.warning(f"Parquet load failed: {parquet_error}")
            raise RuntimeError("Could not load data from Parquet")

        # Display initial data info
        logger.info(f"Loaded {df.count()} rows")
        logger.info("Input schema:")
        df.printSchema()

        # 3. Add calculated features (NO SCALING)
        logger.info("Starting feature engineering...")
        preprocessor = SparkDataPreprocessor(enable_expensive_features=True)

        # This returns the DF with all original columns + new calculated columns
        df_with_features = preprocessor.transform(df)
        
        logger.info(f"Feature engineering completed.")
        logger.info(f"Added {len(preprocessor.derived_feature_cols)} calculated features")
        logger.info(f"Calculated features: {preprocessor.derived_feature_cols}")

        # 4. Save feature metadata for reference
        feature_metadata = {
            'calculated_features': preprocessor.derived_feature_cols,
            'total_calculated_features': len(preprocessor.derived_feature_cols)
        }
        
        metadata_path = PROCESSED_DATA_PATH.parent / "feature_metadata.json"
        os.makedirs(str(PROCESSED_DATA_PATH.parent), exist_ok=True)
        
        with open(str(metadata_path), 'w') as f:
            import json
            json.dump(feature_metadata, f, indent=2)
        logger.info(f"Feature metadata saved to {metadata_path}")

        # 5. Save the Processed Data for Feast
        logger.info(f"Writing processed data to {PROCESSED_DATA_PATH}...")
        df_with_features.write \
            .mode("overwrite") \
            .parquet(str(PROCESSED_DATA_PATH))
            
        logger.info(f"✅ Processed data saved to {PROCESSED_DATA_PATH}")
        
        # 6. Verification
        logger.info("\n--- VERIFICATION ---")
        logger.info(f"Output row count: {df_with_features.count()}")
        logger.info("\nFinal schema:")
        df_with_features.printSchema()
        logger.info("\nSample of output (showing all columns, first 3 rows):")
        df_with_features.show(3, truncate=True)
        
        logger.info("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Output contains all original columns + {len(preprocessor.derived_feature_cols)} calculated features")
        logger.info("Data is ready for Feast ingestion")

    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    generate()