import logging
import os
import glob
from pyspark.sql import SparkSession, DataFrame
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for data loading using PySpark"""
    def __init__(self, spark: SparkSession, data_dir: Path):
        """
        Args:
            spark: Active SparkSession
            data_dir: Path to the directory containing data
        """
        self.spark = spark
        self.data_dir = str(data_dir) # Spark expects string paths

    def _inspect_directory(self, full_path: str):
        """
        Internal debug method to list files in the target directory 
        using standard Python I/O before Spark touches it.
        """
        # Extract the directory part from the wildcard path (e.g. /data/*.parquet -> /data)
        directory = os.path.dirname(full_path)
        pattern = os.path.basename(full_path)
        
        logger.info(f"--- DEBUG: Inspecting directory: {directory} ---")
        
        if not os.path.exists(directory):
            logger.error(f"❌ DIRECTORY DOES NOT EXIST: {directory}")
            logger.error(f"Current working directory is: {os.getcwd()}")
            # List contents of the parent to see where we are
            parent = os.path.dirname(directory)
            if os.path.exists(parent):
                logger.info(f"Contents of parent ({parent}): {os.listdir(parent)}")
            return

        # List all files in that directory
        all_files = os.listdir(directory)
        logger.info(f"Files found in directory ({len(all_files)} total): {all_files[:10]} ...")
        
        # Check specific pattern match
        matched_files = glob.glob(full_path)
        logger.info(f"Files matching pattern '{pattern}': {len(matched_files)}")
        
        if len(matched_files) == 0:
            logger.warning(f"⚠️ Directory exists but NO files match pattern: {pattern}")
        
        logger.info("--- DEBUG END ---")

    def load_data(self, file_pattern: str = "train_set.parquet", file_format: str = "parquet") -> DataFrame:
        """
        Loads data from the directory using Spark's distributed reader.
        
        Args:
            file_pattern: Pattern for files (e.g., "*.parquet")
            file_format: format specifier for spark.read (parquet, csv, etc.)
        Returns:
            DataFrame: Spark DataFrame
        """
        # Construct the full path (e.g., /hist_ingestion/data/historical_data/*.csv)
        full_path = os.path.join(self.data_dir, file_pattern)
        
        # --- RUN DEBUG INSPECTION ---
        self._inspect_directory(full_path)
        # ----------------------------

        logger.info(f"Attempting to load data from: {full_path}")

        try:
            # Spark handles wildcards (*) and multiple files automatically
            df = self.spark.read.format(file_format) \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .load(full_path)
            
            # Action to count rows (triggers computation)
            count = df.count()
            logger.info(f"Successfully loaded data. Total rows: {count}")
            
            if count == 0:
                raise ValueError(f"No data found in {full_path}")
                
            return df

        except Exception as e:
            logger.error(f"Error loading data with Spark: {e}")
            raise