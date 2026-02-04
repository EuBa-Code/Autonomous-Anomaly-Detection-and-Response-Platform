import logging
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

    def load_data(self, file_pattern: str = "*.parquet", file_format: str = "parquet") -> DataFrame:
        """
        Loads data from the directory using Spark's distributed reader.
        
        Args:
            file_pattern: Pattern for files (e.g., "*.parquet")
            file_format: format specifier for spark.read (parquet, csv, etc.)
        Returns:
            DataFrame: Spark DataFrame
        """
        full_path = f"{self.data_dir}/{file_pattern}"
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