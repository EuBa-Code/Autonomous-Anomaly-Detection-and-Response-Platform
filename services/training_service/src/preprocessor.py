"""
Preprocessor module for feature scaling and transformation.

Wraps a StandardScaler to normalize sensor features before model training
and inference. The fitted preprocessor is saved as a joblib artifact
so the streaming service can apply the same transformation at inference time.
"""
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import config

logger = logging.getLogger(__name__)


# Columns that are metadata/labels — must NOT be scaled
NON_FEATURE_COLUMNS = config.LABEL_COLUMNS


class Preprocessor:
    """
    Fits a StandardScaler on training data, transforms new data,
    and persists itself as a joblib artifact.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        """
        Args:
            feature_columns: Explicit list of columns to scale.
                             If None, inferred at fit time by dropping NON_FEATURE_COLUMNS.
        """
        self.scaler = StandardScaler()
        self.feature_columns = feature_columns
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        Fit the scaler on training data.

        Args:
            df: Raw training DataFrame (may contain label/meta columns).

        Returns:
            self (for method chaining)
        """
        if self.feature_columns is None:
            self.feature_columns = [
                col for col in df.columns if col not in NON_FEATURE_COLUMNS
            ]

        logger.info(f"Fitting preprocessor on {len(self.feature_columns)} features")
        logger.info(f"  Features: {self.feature_columns}")

        X = df[self.feature_columns].astype(np.float64)
        self.scaler.fit(X)
        self.is_fitted = True

        logger.info("✓ Preprocessor fitted successfully")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted scaler to new data.

        Args:
            df: DataFrame with at least the feature columns.

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        X = df[self.feature_columns].astype(np.float64)
        return self.scaler.transform(X)

    def preprocess_data(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """
        High-level helper used by the training pipeline.

        Args:
            df:  Raw DataFrame.
            fit: If True, fit the scaler first (training); if False, only transform.

        Returns:
            pd.DataFrame with scaled feature columns.
        """
        if fit:
            self.fit(df)

        scaled = self.transform(df)
        return pd.DataFrame(scaled, columns=self.feature_columns, index=df.index)

    def save(self, filepath: Path) -> None:
        """Persist the fitted preprocessor to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"✓ Preprocessor saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "Preprocessor":
        """Load a previously saved preprocessor."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found: {filepath}")

        preprocessor = joblib.load(filepath)
        logger.info(f"✓ Preprocessor loaded from: {filepath}")
        return preprocessor
