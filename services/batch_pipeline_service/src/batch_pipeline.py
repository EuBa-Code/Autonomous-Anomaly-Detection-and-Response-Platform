import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from data_engineering_service.data_engineering import FeatureEngineering


# --- SETTINGS ---
@dataclass(frozen=True)
class Settings:
    """Centralized configuration for Washing Machine Batch Pipeline"""
    historical_dir: str           # Path to raw data lake
    offline_dir: str              # Path to offline feature store
    feature_config_path: str      # Path to feature config YAML
    spark_app_name: str           # Spark application name
    spark_timezone: str           # Spark session timezone
    spark_partitions: int         # Number of output partitions
    
    # Feature column mappings
    machine_id_col: str
    timestamp_col: str
    timestamp_grain: str          # "day", "hour", or "minute"
    target_feature_col: str
    output_timestamp_col: str
    
    # Processing options
    output_feature_name: str      # Feature store folder name
    write_mode: str               # "append" or "overwrite"
    allow_null_values: bool
    show_sample_rows: int


def load_settings() -> Settings:
    """Load configuration from environment variables with defaults"""
    return Settings(
        historical_dir=os.getenv("HISTORICAL_DIR", "/app/data/historical_data"),
        offline_dir=os.getenv("OFFLINE_DIR", "/app/data/feature_store"),
        feature_config_path=os.getenv("FEATURE_CONFIG_PATH", "config/feature_config.yaml"),
        spark_app_name=os.getenv("SPARK_APP_NAME", "WashingMachine-Batch-Append-Clean"),
        spark_timezone=os.getenv("SPARK_TIMEZONE", "UTC"),
        spark_partitions=int(os.getenv("SPARK_PARTITIONS", "1")),
        machine_id_col=os.getenv("MACHINE_ID_COL", "Machine_ID"),
        timestamp_col=os.getenv("TIMESTAMP_COL", "timestamp"),
        timestamp_grain=os.getenv("TIMESTAMP_GRAIN", "day"),
        target_feature_col=os.getenv("TARGET_FEATURE_COL", "Daily_Vibration_PeakMean_Ratio"),
        output_timestamp_col=os.getenv("OUTPUT_TIMESTAMP_COL", "event_timestamp"),
        output_feature_name=os.getenv("OUTPUT_FEATURE_NAME", "machine_batch_features"),
        write_mode=os.getenv("WRITE_MODE", "append"),
        allow_null_values=os.getenv("ALLOW_NULL_VALUES", "false").lower() == "true",
        show_sample_rows=int(os.getenv("SHOW_SAMPLE_ROWS", "5")),
    )


# --- HELPER FUNCTIONS ---

def _read_parquet(spark: SparkSession, path: Path) -> DataFrame:
    """
    Read parquet file(s) into a Spark DataFrame with proper error handling.
    
    Args:
        spark: SparkSession instance
        path: Path to parquet file or directory
        
    Returns:
        DataFrame: Loaded parquet data
        
    Raises:
        FileNotFoundError: If parquet path does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")
    
    print(f"[*] Reading parquet from: {path}")
    df = spark.read.parquet(str(path))
    row_count = df.count()
    print(f"[*] Loaded {row_count} rows")
    return df


def validate_settings(settings: Settings) -> bool:
    """
    Validate critical configuration paths and settings.
    
    Args:
        settings: Settings instance to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    historical_path = Path(settings.historical_dir)
    
    if not historical_path.exists():
        print(f"[!] Warning: HISTORICAL_DIR does not exist: {historical_path}")
        return False
    
    if not historical_path.is_dir():
        print(f"[!] Error: HISTORICAL_DIR is not a directory: {historical_path}")
        return False
    
    # Check if there are any parquet files
    parquet_files = list(historical_path.glob("*.parquet"))
    if not parquet_files:
        print(f"[!] Warning: No parquet files found in: {historical_path}")
        return False
    
    print(f"[✓] Validation passed. Found {len(parquet_files)} parquet files")
    return True


def apply_feature_engineering(
    raw_df: DataFrame, 
    settings: Settings
) -> DataFrame:
    """
    Apply feature engineering transformations to raw data.
    
    Args:
        raw_df: Raw input DataFrame
        settings: Settings instance with configuration
        
    Returns:
        DataFrame: Feature-engineered DataFrame
    """
    print("[*] Executing feature engineering...")
    
    engineer = FeatureEngineering(config_path=settings.feature_config_path)
    enriched_df = engineer._apply_batch_features(raw_df)
    
    return enriched_df


def clean_and_prepare_features(
    enriched_df: DataFrame,
    settings: Settings
) -> DataFrame:
    """
    Clean schema, handle nulls, remove duplicates.
    
    Args:
        enriched_df: Feature-engineered DataFrame
        settings: Settings instance with configuration
        
    Returns:
        DataFrame: Cleaned and prepared DataFrame
    """
    print("[*] Cleaning schema and removing duplicates...")
    
    # Select and truncate timestamp to desired grain
    final_df = enriched_df.select(
        settings.machine_id_col,
        F.date_trunc(
            settings.timestamp_grain,
            F.col(settings.timestamp_col)
        ).alias(settings.output_timestamp_col),
        settings.target_feature_col,
    ).distinct()
    
    # Filter null values if configured
    if not settings.allow_null_values:
        print("[*] Filtering null values...")
        final_df = final_df.filter(
            F.col(settings.target_feature_col).isNotNull()
        )
    
    row_count = final_df.count()
    print(f"[*] Rows after cleaning: {row_count}")
    
    return final_df


def write_offline(
    df: DataFrame,
    output_path: Path,
    write_mode: str,
    num_partitions: int
) -> None:
    """
    Write DataFrame to offline feature store (parquet format).
    
    Args:
        df: DataFrame to write
        output_path: Destination path in feature store
        write_mode: Write mode ("append", "overwrite", etc.)
        num_partitions: Number of output partitions
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Writing features to {output_path}")
    print(f"[*] Write mode: {write_mode}, Partitions: {num_partitions}")
    
    (
        df
        .repartition(num_partitions)
        .write
        .mode(write_mode)
        .parquet(str(output_path))
    )
    
    print(f"[✓] Successfully wrote {df.count()} rows")


def show_sample_output(df: DataFrame, num_rows: int = 5) -> None:
    """Display sample output for verification"""
    print(f"[*] Sample output (first {num_rows} rows):")
    df.show(num_rows, truncate=False)


# --- MAIN PIPELINE ---

def run_batch_pipeline() -> None:
    """
    Main batch pipeline for washing machine feature engineering.
    
    Workflow:
    1. Load configuration and validate
    2. Initialize Spark Session
    3. Load raw data from historical data lake
    4. Apply feature engineering transformations
    5. Clean schema, remove duplicates, handle nulls
    6. Write features to offline feature store
    7. Display sample output
    """
    
    # 1. Load and validate configuration
    print("[*] Loading configuration...")
    settings = load_settings()
    
    if not validate_settings(settings):
        print("[!] Configuration validation failed. Aborting pipeline.")
        return
    
    # 2. Initialize Spark Session
    print("[*] Initializing Spark Session...")
    spark = (
        SparkSession.builder
        .appName(settings.spark_app_name)
        .config("spark.sql.session.timeZone", settings.spark_timezone)
        .getOrCreate()
    )
    
    try:
        # 3. Load raw data
        historical_path = Path(settings.historical_dir)
        raw_df = _read_parquet(spark, historical_path)
        
        # 4. Apply feature engineering
        enriched_df = apply_feature_engineering(raw_df, settings)
        
        # 5. Clean and prepare features
        final_df = clean_and_prepare_features(enriched_df, settings)
        
        # 6. Write to feature store
        output_path = Path(settings.offline_dir) / settings.output_feature_name
        write_offline(
            final_df,
            output_path,
            settings.write_mode,
            settings.spark_partitions
        )
        
        # 7. Show sample output
        show_sample_output(final_df, settings.show_sample_rows)
        
        # Log completion
        end_time = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
        print(f"[✓] Pipeline completed successfully at {end_time}")
        print(f"[✓] Features written to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"[!] File not found error: {e}")
        raise
    except Exception as e:
        print(f"[!] Error during pipeline execution: {e}")
        raise
    finally:
        print("[*] Stopping Spark Session...")
        spark.stop()


if __name__ == "__main__":
    run_batch_pipeline()