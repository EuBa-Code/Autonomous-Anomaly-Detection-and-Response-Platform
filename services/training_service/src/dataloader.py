import logging
from pathlib import Path
from typing import Optional, List
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

logger=logging.getLogger(__name__)  # logger associated with the current module/file

class DataLoader:
    """Class for data loading and preprocessing"""
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.feature_columns: Optional[List[str]] = None  # Stores the columns used for fitting
        self.label_columns = ['Is_Anomaly', 'Anomaly_Type'] # label columns to exclude in preprocessing (present only in test set)

    def load_data(self, filename: str = "*.csv") -> pd.DataFrame:
        """
        Loads data from the directory
        Args:
            filename: Pattern for files to load
        Returns:
            pd.DataFrame: DataFrame with all loaded data
        """
        data_files=list(self.data_dir.glob(filename))

        if not data_files:
            raise FileNotFoundError (f"No file found with pattern {filename} in {self.data_dir}")
        
        logger.info(f"Loading {len(data_files)} files...")

        dfs= []
        for file_path in data_files:
            try:
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path, sep=None, engine="python", encoding_errors="replace")  # auto-detect separator,  more tolerant parser, avoids encoding crashes
                elif file_path.suffix == ".parquet":
                    df= pd.read_parquet(file_path)    
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                logger.info(f" Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        if not dfs:
            raise ValueError("No data was loaded successfully")
        
        # Concatenate all DataFrames
        data=pd.concat(dfs, ignore_index= True)
        logger.info(f"Total data loaded: {len(data)} rows, {len(data.columns)} columns")

        return data