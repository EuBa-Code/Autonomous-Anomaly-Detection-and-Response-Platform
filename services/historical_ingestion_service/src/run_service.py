# generate_preprocessor.py
import logging
import sys
from pathlib import Path
from pyspark.sql import SparkSession
import os

from config import RAW_DATA_PATH, PROCESSED_DATA, TRAIN_LABELS, TRAIN, TEST

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
            df_train = loader.load_data(file_pattern="train_set.parquet", file_format="parquet")
            df_test = loader.load_data(file_pattern="test_set.parquet", file_format='parquet')
            df_train_labels = loader.load_data(file_pattern="train_set_labels.parquet", file_format="parquet")
            
        except Exception as parquet_error:
            logger.warning(f"Parquet load failed: {parquet_error}")
            raise RuntimeError("Could not load data from Parquet")

        # Display initial data info
        logger.info(f"Loaded {df_train_labels.count()} rows")
        logger.info("Input schema (train with labels):")
        df_train_labels.printSchema()

        # 3. Add calculated features (NO SCALING)
        logger.info("Starting feature engineering...")
        preprocessor = SparkDataPreprocessor(enable_expensive_features=False)

        # This returns the DF with all original columns + new calculated columns
        df_train_features = preprocessor.transform(df_train)
        df_train_labels_features = preprocessor.transform(df_train_labels)
        df_test_features = preprocessor.transform(df_test)

        logger.info(f"Feature engineering completed.")
        logger.info(f"Added {len(preprocessor.derived_feature_cols)} calculated features")
        logger.info(f"Calculated features: {preprocessor.derived_feature_cols}")

        # 4. Save feature metadata for reference
        feature_metadata = {
            'calculated_features': preprocessor.derived_feature_cols,
            'total_calculated_features': len(preprocessor.derived_feature_cols)
        }
        
        metadata_path = PROCESSED_DATA.parent / "pyspark_metadata" / "feature_metadata.json"
        os.makedirs(str(metadata_path.parent), exist_ok=True)
        
        with open(str(metadata_path), 'w') as f:
            import json
            json.dump(feature_metadata, f, indent=2)
        logger.info(f"Feature metadata saved to {metadata_path}")

        # 5. Save the Processed Data for Feast
        logger.info(f"Writing processed data train to {TRAIN}...")
        df_train_features \
            .orderBy("timestamp", "Machine_ID") \
            .coalesce(1) \
            .write \
            .mode("overwrite") \
            .parquet(str(TRAIN))

        logger.info(f"Writing processed data train labels to {TRAIN_LABELS}...")
        df_train_labels_features \
            .orderBy("timestamp", "Machine_ID") \
            .coalesce(1) \
            .write \
            .mode("overwrite") \
            .parquet(str(TRAIN_LABELS))
            
        logger.info(f"Writing processed data test to {TEST}...")
        df_test_features \
            .orderBy("timestamp", "Machine_ID") \
            .coalesce(1) \
            .write \
            .mode("overwrite") \
            .parquet(str(TEST))
                    
        logger.info(f"✅ Processed data saved!")
        
        # 6. Verification
        logger.info("\n--- VERIFICATION (train with labels) ---")
        logger.info(f"Output row count: {df_train_labels_features.count()}")
        logger.info("\nFinal schema:")
        df_train_labels_features.printSchema()
        logger.info("\nSample of output (showing all columns, first 3 rows):")
        df_train_labels_features.show(3, truncate=True)
        
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