import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Centralized configuration for the Washing Machine Batch Pipeline"""

    # Spark Configuration
    SPARK_APP_NAME = "WashingMachine-Batch-Append-Clean"
    SPARK_TIMEZONE = "UTC"
    SPARK_CONFIG: Dict[str, str] = {
        "spark.sql.session.timeZone": SPARK_TIMEZONE,
    }

    # Data Paths
    HISTORICAL_DIR = os.getenv("HISTORICAL_DIR", "/app/data/historical_data")
    OFFLINE_DIR = os.getenv("OFFLINE_DIR", "/app/data/feature_store")
    OUTPUT_FEATURE_NAME = "machine_batch_features"
    OUTPUT_PATH = Path(OFFLINE_DIR) / OUTPUT_FEATURE_NAME

    # Feature Engineering
    FEATURE_CONFIG_PATH = os.getenv("FEATURE_CONFIG_PATH", "config/feature_config.yaml")

    # Processing Configuration
    OUTPUT_REPARTITION = 1
    WRITE_MODE = "append"  # Options: "overwrite", "append", "ignore", "error"
    LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))

    # Feature Column Names
    MACHINE_ID_COL = "Machine_ID"
    TIMESTAMP_COL = "timestamp"
    TIMESTAMP_GRAIN = "day"  # Options: "day", "hour", "minute"
    TARGET_FEATURE_COL = "Daily_Vibration_PeakMean_Ratio"
    OUTPUT_TIMESTAMP_COL = "event_timestamp"

    # Output Schema
    OUTPUT_COLUMNS = [
        MACHINE_ID_COL,
        OUTPUT_TIMESTAMP_COL,
        TARGET_FEATURE_COL,
    ]

    # Quality Checks
    ALLOW_NULL_VALUES = False

    # Logging
    SHOW_SAMPLE_ROWS = 5

    # Partitioning
    PARTITION_COLS = ["p_date"]

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "spark_app_name": cls.SPARK_APP_NAME,
            "spark_timezone": cls.SPARK_TIMEZONE,
            "historical_dir": cls.HISTORICAL_DIR,
            "offline_dir": cls.OFFLINE_DIR,
            "output_path": str(cls.OUTPUT_PATH),
            "feature_config_path": cls.FEATURE_CONFIG_PATH,
            "write_mode": cls.WRITE_MODE,
            "lookback_days": cls.LOOKBACK_DAYS,
            "output_columns": cls.OUTPUT_COLUMNS,
            "allow_null_values": cls.ALLOW_NULL_VALUES,
        }

    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration paths"""
        try:
            historical_path = Path(cls.HISTORICAL_DIR)
            if not historical_path.exists():
                print(f"[!] Warning: HISTORICAL_DIR does not exist: {cls.HISTORICAL_DIR}")
                return False
            
            if not historical_path.is_dir():
                print(f"[!] Error: HISTORICAL_DIR is not a directory: {cls.HISTORICAL_DIR}")
                return False
            
            # Check for parquet files
            parquet_files = list(historical_path.glob("*.parquet"))
            if not parquet_files:
                print(f"[!] Warning: No parquet files found in: {cls.HISTORICAL_DIR}")
                return False
            
            return True
        except Exception as e:
            print(f"[!] Configuration validation error: {e}")
            return False