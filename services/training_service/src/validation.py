"""
Module for configuration and path validation.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validator for model configuration"""
    
    def __init__(self, data_dir: Path, model_dir: Path, metrics_dir: Path):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.metrics_dir = Path(metrics_dir)
    
    def validate_all(self) -> Tuple[bool, Optional[str]]:
        """
        Validates all required paths
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not self._validate_data_directory():
            return False, f"Invalid data directory: {self.data_dir}"
        
        if not self._ensure_directory_exists(self.model_dir):
            return False, f"Unable to create model directory: {self.model_dir}"
        
        if not self._ensure_directory_exists(self.metrics_dir):
            return False, f"Unable to create metrics directory: {self.metrics_dir}"
        
        logger.info("Configuration validation completed successfully")
        return True, None
    
    def _validate_data_directory(self) -> bool:
        """Checks that the data directory exists and contains files"""
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return False
        
        if not self.data_dir.is_dir():
            logger.error(f"Path is not a directory: {self.data_dir}")
            return False
        
        data_files = list(self.data_dir.glob("*.csv")) + list(self.data_dir.glob("*.parquet"))
        if not data_files:
            logger.error(f"No data files found in: {self.data_dir}")
            return False
        
        logger.info(f"Found {len(data_files)} data files in {self.data_dir}")
        return True
     
    def _ensure_directory_exists(self, directory: Path) -> bool:
        """Creates a directory if it does not exist"""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory verified/created: {directory}")
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False
    
    def get_data_files(self, extensions: list = None) -> list:
        """
        Returns the list of data files
        
        Args:
            extensions: List of extensions to look for (default: ['.csv', '.parquet'])
        
        Returns:
            list: List of Paths to data files
        """
        if extensions is None:
            extensions = ['.csv', '.parquet']
        
        data_files = []
        for ext in extensions:
            data_files.extend(self.data_dir.glob(f"*{ext}"))
        
        return sorted(data_files)