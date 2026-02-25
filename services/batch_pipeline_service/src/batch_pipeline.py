"""
Batch Feature Pipeline for Washing Machine Anomaly Detection

This script computes daily batch features for the Feast feature store:
  • Daily_Vibration_PeakMean_Ratio: Daily max(Vibration) / mean(Vibration) per machine
  
The pipeline:
  1. Loads configuration from YAML file
  2. Reads processed industrial washer data from the data warehouse
  3. Computes daily aggregations grouped by machine and calendar day
  4. Joins aggregations back to each row (feature enrichment)
  5. Writes results to offline store in Parquet format (Feast-compatible)

Architecture:
  - Batch features have daily/weekly TTL (vs streaming features with hourly TTL)
  - Separate feature view allows independent refresh cadences
  - Both views share Machine_ID entity for unified feature retrieval at inference time
"""

import os
import yaml
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import (
    DataFrame,
    SparkSession,
    Window
)
from pyspark.sql import functions as F

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SETTINGS DATACLASS
# ============================================================================

@dataclass(frozen=True)
class Settings:
    """Configuration for the batch feature pipeline"""
    data_warehouse_dir: str  # Path to input data (processed industrial washer features)
    offline_store_dir: str   # Path to output offline store (Feast-compatible)
    spark_partitions: int    # Number of Spark partitions for parallelization
    spark_master: str        # Spark master URL
    spark_app_name: str      # Application name for Spark
    timestamp_column: str    # Name of timestamp column
    write_mode: str          # Write mode: 'overwrite', 'append', etc.


def load_settings(config_path: str) -> Settings:
    """
    Load configuration from YAML file and create Settings dataclass
    
    Args:
        config_path: Path to batch_config.yaml
        
    Returns:
        Settings instance with all configuration loaded
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract configurations with sensible defaults
    paths = config.get('paths', {})
    spark_cfg = config.get('spark', {})
    schema = config.get('schema', {})
    processing = config.get('processing', {})
    
    return Settings(
        data_warehouse_dir=paths.get('data_warehouse_dir', 'data/processed_datasets/industrial_washer_normal_features'),
        offline_store_dir=paths.get('offline_store_dir', 'data/offline/machine_batch_features'),
        spark_partitions=spark_cfg.get('partitions', 8),
        spark_master=spark_cfg.get('master', 'local[*]'),
        spark_app_name=spark_cfg.get('app_name', 'batch-feature-pipeline'),
        timestamp_column=schema.get('timestamp_column', 'timestamp'),
        write_mode=processing.get('write_mode', 'overwrite'),
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _read_parquet(spark: SparkSession, path: Path) -> DataFrame:
    """
    Read Parquet file(s) from path
    
    Args:
        spark: SparkSession instance
        path: Path to parquet file or directory (supports PySpark partitioned structure)
        
    Returns:
        Spark DataFrame
        
    Raises:
        FileNotFoundError: If path does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {path}")
    
    logger.info(f"Reading parquet from: {path}")
    df = spark.read.parquet(str(path))
    logger.info(f"Loaded {df.count()} rows")
    return df


def read_inputs(spark: SparkSession, data_warehouse_dir: str) -> DataFrame:
    """
    Read processed industrial washer data from data warehouse
    
    Args:
        spark: SparkSession instance
        data_warehouse_dir: Path to processed features directory
        
    Returns:
        Spark DataFrame with columns:
          - Machine_ID
          - timestamp
          - Vibration_mm_s (and other sensor columns)
    """
    root = Path(data_warehouse_dir)
    
    # Read the main processed features parquet dataset
    # Expected to have multiple part-*.parquet files in partitioned structure
    df = _read_parquet(spark, root)
    
    logger.info(f"Schema of input data:")
    df.printSchema()
    
    return df


def compute_daily_batch_features(df: DataFrame, timestamp_col: str) -> DataFrame:
    """
    Compute daily batch features grouped by Machine_ID and calendar day
    
    Formula: Daily_Vibration_PeakMean_Ratio = max(Vibration_mm_s) / mean(Vibration_mm_s)
    
    Interpretation:
      - High ratio: spiky/impulsive vibration → potential mechanical fault
      - Low ratio: smooth vibration → healthy operation
      
    Implementation approach (matching data_engineering.py):
      1. Create daily window using F.window("timestamp", "1 day")
      2. Group by Machine_ID and window period
      3. Compute max/mean aggregation
      4. Join result back to every row within that day (enrichment)
      5. Null out values for incomplete first period per machine
    
    Args:
        df: Input DataFrame with timestamp and Vibration_mm_s columns
        timestamp_col: Name of timestamp column
        
    Returns:
        DataFrame with original columns + Daily_Vibration_PeakMean_Ratio
    """
    logger.info("Computing daily batch features")
    
    # Cast timestamp to ensure it's in proper format for windowing
    df = df.withColumn(timestamp_col, F.col(timestamp_col).cast("timestamp"))
    
    # Step 1: Create daily tumbling window
    # Groups each row into the start of the day it belongs to
    # Example: 2024-01-15 06:30:45 → 2024-01-15 00:00:00 (window start)
    logger.info("Creating daily window aggregation...")
    
    daily_window = (
        df.groupBy(
            F.col("Machine_ID"),
            F.window(timestamp_col, "1 day").alias("window")
        )
        .agg(
            # Daily peak-to-mean ratio: max(Vibration) / mean(Vibration)
            (F.max("Vibration_mm_s") / F.mean("Vibration_mm_s"))
            .alias("Daily_Vibration_PeakMean_Ratio"),
            
            # Keep track of the latest timestamp in each window (for Feast FileSource requirement)
            F.max(timestamp_col).alias(timestamp_col),
        )
    )
    
    logger.info("Daily aggregation computed")
    
    # Step 2: For each machine, identify the latest (most recent) daily window
    # This is needed because we only care about the current day's aggregation value
    # to join back to rows
    logger.info("Selecting latest daily window per machine...")
    
    w = Window.partitionBy("Machine_ID").orderBy(F.col("window.end").desc())
    daily_latest = (
        daily_window
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .select(
            "Machine_ID",
            "Daily_Vibration_PeakMean_Ratio",
            timestamp_col
        )
    )
    
    # Step 3: Join daily aggregation back to each row
    # This enriches every row with the daily aggregate for its machine+day
    logger.info("Joining daily features back to original data...")
    
    # For proper joining, we need to align timestamps to day boundaries
    df_with_daily = (
        df
        .withColumn("_day", F.date_trunc("day", F.col(timestamp_col)))
        .join(
            daily_window
            .withColumn("_day", F.date_trunc("day", F.col("window.end"))),
            on=["Machine_ID", "_day"],
            how="left"
        )
        .drop("_day", "window")
    )
    
    logger.info("Daily features joined successfully")
    
    # Step 4: Handle incomplete periods
    # The first incomplete period per machine (e.g., if data starts at 06:00)
    # should have NULL for daily features since aggregation is incomplete
    # This prevents misleading feature values based on partial day data
    logger.info("Applying first-period guard (nulling incomplete initial period)...")
    
    _FIRST_PERIOD_COL = "_batch_first_period_"
    machine_window = Window.partitionBy("Machine_ID")
    
    df_with_daily = (
        df_with_daily
        .withColumn(
            "_day_period",
            F.date_trunc("day", F.col(timestamp_col))
        )
        .withColumn(
            _FIRST_PERIOD_COL,
            F.min("_day_period").over(machine_window)
        )
        .withColumn(
            "Daily_Vibration_PeakMean_Ratio",
            F.when(
                F.col("_day_period") == F.col(_FIRST_PERIOD_COL),
                F.lit(None).cast("double")  # First period → null
            ).otherwise(F.col("Daily_Vibration_PeakMean_Ratio"))  # Others → keep value
        )
        .drop("_day_period", _FIRST_PERIOD_COL)
    )
    
    logger.info("✓ Batch feature computation complete")
    
    return df_with_daily


def write_offline(df: DataFrame, out_path: Path, partitions: int, write_mode: str) -> None:
    """
    Write feature DataFrame to offline store in Parquet format
    
    Output format: Multiple part-*.parquet files (PySpark partitioned structure)
    This matches Feast's FileSource expectations
    
    Args:
        df: DataFrame to write
        out_path: Output directory path
        partitions: Number of output partitions (files)
        write_mode: Spark write mode ('overwrite', 'append', etc.)
    """
    logger.info(f"Writing offline features to: {out_path}")
    logger.info(f"Write mode: {write_mode}")
    logger.info(f"Repartitioning to {partitions} output files")
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Repartition for even distribution across output files
    # Then write in Parquet format
    (
        df
        .repartition(partitions)
        .write
        .mode(write_mode)
        .parquet(str(out_path))
    )
    
    logger.info(f"✓ Data written successfully to {out_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Main orchestration function for batch feature pipeline"""
    
    logger.info("=" * 80)
    logger.info("BATCH FEATURE PIPELINE - WASHING MACHINE ANOMALY DETECTION")
    logger.info("=" * 80)
    
    # 1. Load configuration
    config_path = "batch_config.yaml"
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please create batch_config.yaml in the current directory")
        return
    
    settings = load_settings(config_path)
    logger.info(f"Configuration loaded: {config_path}")
    
    # 2. Initialize Spark session
    logger.info("Initializing Spark session...")
    spark = (
        SparkSession.builder
        .appName(settings.spark_app_name)
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(max(8, settings.spark_partitions * 2)))
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session initialized")
    
    try:
        # 3. Read input data from data warehouse
        logger.info("Reading input data from data warehouse...")
        df = read_inputs(spark, settings.data_warehouse_dir)
        
        # 4. Compute daily batch features
        logger.info("Computing daily batch features...")
        df_with_features = compute_daily_batch_features(df, settings.timestamp_column)
        
        # 5. Show sample output for verification
        logger.info("Sample output (first 10 rows):")
        df_with_features.select(
            "Machine_ID",
            settings.timestamp_column,
            "Vibration_mm_s",
            "Daily_Vibration_PeakMean_Ratio"
        ).show(10, truncate=False)
        
        # 6. Write to offline store
        offline_path = Path(settings.offline_store_dir)
        write_offline(
            df_with_features,
            offline_path,
            settings.spark_partitions,
            settings.write_mode
        )
        
        # 7. Print summary
        logger.info("=" * 80)
        logger.info("BATCH PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Output location: {offline_path}")
        logger.info(f"Features computed: Daily_Vibration_PeakMean_Ratio")
        
        # Suggested timestamp for Feast materialize-incremental command
        end_date = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
        logger.info(f"Suggested end-date for 'feast materialize-incremental' (UTC): {end_date}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # 8. Clean up
        logger.info("Stopping Spark session...")
        spark.stop()
        logger.info("Pipeline finished")


if __name__ == "__main__":
    main()