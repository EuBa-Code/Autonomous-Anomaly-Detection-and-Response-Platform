# Feature Engineering with QuixStreams
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging (basic configuration)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, label_columns: List[str], scaler_type: str = 'standard'):
        """
        Initializes the preprocessor.
        
        Args:
            label_columns: List of column names to exclude from scaling (target variables).
            scaler_type: Type of scaler to use ('standard' or 'minmax').
        """
        self.label_columns = label_columns
        self.feature_columns: Optional[List[str]] = None
        
        # Initialize the specific scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

    def preprocess_data(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocesses data for training by scaling numeric features.

        Args:
            data: DataFrame with raw data.
            fit: If True, computes the mean and std to be used for later scaling.
                 If False, uses previously computed values.

        Returns:
            features (X): DataFrame with scaled features.
        """
        logger.info("Starting data preprocessing...")
        
        df = data.copy()

        # 1. Identify all numeric columns
        all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 2. EXPLICITLY EXCLUDE label columns from columns to scale
        # Use set operations for safety if a label column is not numeric
        cols_to_scale = [c for c in all_numeric_cols if c not in self.label_columns]

        if not cols_to_scale:
            raise ValueError("No valid numeric columns to scale found")

        if fit:
            # Save exactly which columns we scaled
            self.feature_columns = cols_to_scale
            # Fit and transform
            X_scaled = self.scaler.fit_transform(df[self.feature_columns])
            logger.info(f"Normalization completed on {len(self.feature_columns)} columns")
        else:
            # Check if fitted
            if self.feature_columns is None:
                raise ValueError("Scaler not fitted. Run with fit=True first.")
                
            # During testing, use ONLY the columns saved during fit
            # Ensure the input data actually has these columns
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Input data is missing columns required by the scaler: {missing_cols}")

            X_scaled = self.scaler.transform(df[self.feature_columns])
            logger.info("Normalization (transform) completed")

        # Return as DataFrame with column names
        X = pd.DataFrame(X_scaled, 
                         columns=self.feature_columns, 
                         index=df.index)

        return X
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs inverse transform on normalized data.

        Args:
            data: DataFrame with normalized data.

        Returns:
            DataFrame with data in the original scale.
        """
        if self.feature_columns is None:
            raise ValueError("Scaler not yet fitted. Run preprocess_data with fit=True first")
        
        # Check that columns match
        if list(data.columns) != self.feature_columns:
            raise ValueError(
                f"DataFrame columns do not match scaler columns. "
                f"Expected: {self.feature_columns}, Received: {list(data.columns)}"
            )
        
        # Inverse transform
        X_original = self.scaler.inverse_transform(data)
        
        # Return as DataFrame
        result = pd.DataFrame(X_original, columns=self.feature_columns, index=data.index)
        logger.info("Inverse transform completed")
        
        return result

    def save_scaler(self, filepath: Path):
        """
        Saves the scaler for future use (e.g., on test dataset)
        without refitting and without recomputing feature columns.
        """
        try:
            # Ensure the directory exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save both scaler and feature_columns
            scaler_data = {
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(scaler_data, filepath)
            logger.info(f"Scaler and feature columns saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
            raise  # re-raise original error, program stops and logs trace

    def load_scaler(self, filepath: Path):
        """
        Loads a saved scaler and the associated feature column names.
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Scaler file not found at {filepath}")

            scaler_data = joblib.load(filepath)
            self.scaler = scaler_data['scaler']
            self.feature_columns = scaler_data['feature_columns']
            
            logger.info(f"Scaler loaded from {filepath}")
            logger.info(f"Feature columns: {self.feature_columns}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise  # re-raise original error, program stops and logs trace