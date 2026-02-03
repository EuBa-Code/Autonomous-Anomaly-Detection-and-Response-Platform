"""
Data loading module for Training Service.
Handles loading of raw data files without preprocessing.
"""
import logging
from pathlib import Path
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data loading operations.
    Does NOT perform preprocessing - that's the preprocessor's job.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        logger.info(f"DataLoader initialized with directory: {self.data_dir}")

    def load_data(self, filename_pattern: str = "*.parquet") -> pd.DataFrame:
        """
        Load data files matching the pattern.
        
        Args:
            filename_pattern: Glob pattern for files to load (e.g., "train*.parquet")
        
        Returns:
            pd.DataFrame: Concatenated raw data
        
        Raises:
            FileNotFoundError: If no files match the pattern
            ValueError: If no data was loaded successfully
        """
        data_files = list(self.data_dir.glob(filename_pattern))

        if not data_files:
            raise FileNotFoundError(
                f"No files found with pattern '{filename_pattern}' in {self.data_dir}"
            )
        
        logger.info(f"Found {len(data_files)} file(s) matching pattern '{filename_pattern}'")

        dataframes = []
        for file_path in data_files:
            try:
                df = self._load_single_file(file_path)
                logger.info(
                    f"Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns"
                )
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        if not dataframes:
            raise ValueError("No data was loaded successfully")
        
        # Concatenate all DataFrames
        data = pd.concat(dataframes, ignore_index=True)
        logger.info(
            f"Total data loaded: {len(data)} rows, {len(data.columns)} columns"
        )

        return data

    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single data file based on its extension.
        
        Args:
            file_path: Path to the file
        
        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            ValueError: If file type is not supported
        """
        if file_path.suffix == ".csv":
            return pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding_errors="replace"
            )
        elif file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def get_available_files(self, pattern: str = "*") -> List[Path]:
        """
        Get list of available files in data directory.
        
        Args:
            pattern: Glob pattern to filter files
        
        Returns:
            List[Path]: List of file paths
        """
        return sorted(self.data_dir.glob(pattern))