import logging
from typing import List
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pathlib import Path

logger = logging.getLogger(__name__)

class SparkDataPreprocessor:
    def __init__(self, label_columns: List[str], scaler_type: str = 'standard'):
        """
        Args:
            label_columns: Columns to EXCLUDE from scaling (IDs, Timestamps, Targets)
            scaler_type: 'standard' or 'minmax'
        """
        self.label_columns = label_columns
        self.scaler_type = scaler_type
        self.model: PipelineModel = None
        self.feature_cols: List[str] = []

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Identifies numeric columns, assembles them, fits the scaler, 
        and transforms the data.
        """
        logger.info("Starting PySpark preprocessing...")

        # 1. Identify Numeric Columns automatically
        # dtypes returns list of (col_name, data_type)
        numeric_types = ['int', 'bigint', 'float', 'double']
        all_cols = df.dtypes
        
        self.feature_cols = [
            name for name, dtype in all_cols 
            if dtype in numeric_types and name not in self.label_columns
        ]

        logger.info(f"Features selected for scaling: {self.feature_cols}")

        if not self.feature_cols:
            raise ValueError("No numeric columns found to scale.")

        # 2. VectorAssembler: Combines all feature cols into a single 'features_vec'
        assembler = VectorAssembler(
            inputCols=self.feature_cols, 
            outputCol="unscaled_features",
            handleInvalid="skip" # or 'keep'/'error' based on needs
        )

        # 3. Define Scaler
        if self.scaler_type != "standard":
            raise ValueError(f"Unsupported scaler_type: {self.scaler_type}. Only 'standard' is allowed.")

        # withMean=True is expensive in Spark (destroys sparsity), use with caution on massive sparse data
        scaler = StandardScaler(
            inputCol="unscaled_features",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        # 4. Build Pipeline
        pipeline = Pipeline(stages=[assembler, scaler])

        # 5. Fit the model (Compute Mean/Std)
        logger.info("Fitting the Spark Pipeline...")
        self.model = pipeline.fit(df)

        # 6. Transform
        logger.info("Transforming data...")
        transformed_df = self.model.transform(df)

        # Optional: Drop the intermediate 'unscaled_features' to save space
        return transformed_df.drop("unscaled_features")

    def save_model(self, path: str):
        """
        Saves the Spark PipelineModel. 
        This folder can be loaded later to transform new data identically.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        logger.info(f"Saving PipelineModel to {path}")
        # overwrite() allows replacing existing models
        self.model.write().overwrite().save(str(path))