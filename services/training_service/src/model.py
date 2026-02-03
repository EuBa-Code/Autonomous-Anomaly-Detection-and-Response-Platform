"""
Isolation Forest model module with artifact management.
"""
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Wrapper class for Isolation Forest anomaly detection model.
    Handles training, prediction, and artifact management.
    """
    
    def __init__(self, **params):
        """
        Initialize the Isolation Forest model.
        
        Args:
            **params: Parameters for sklearn's IsolationForest
                Common parameters:
                - n_estimators: Number of trees (default: 100)
                - contamination: Expected proportion of anomalies (default: 0.1)
                - max_samples: Number of samples to train each tree (default: 'auto')
                - random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(**params)
        self.params = params
        self.is_trained = False
        logger.info(f"IsolationForest initialized with params: {params}")

    def train(self, X: pd.DataFrame) -> None:
        """
        Train the Isolation Forest model.
        
        Args:
            X: Training features (preprocessed DataFrame)
        
        Raises:
            ValueError: If input data is invalid
        """
        if X.empty:
            raise ValueError("Cannot train on empty dataset")
        
        logger.info(
            f"Training model on {X.shape[0]} samples with {X.shape[1]} features"
        )

        try:
            self.model.fit(X)
            self.is_trained = True
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Features for prediction (preprocessed DataFrame)
        
        Returns:
            np.ndarray: Predictions (1 = normal, -1 = anomaly)
        
        Raises:
            RuntimeError: If model is not trained
        """
        self._check_trained()
        
        logger.debug(f"Predicting on {X.shape[0]} samples")
        return self.model.predict(X)
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Args:
            X: Features (preprocessed DataFrame)
        
        Returns:
            np.ndarray: Anomaly scores (more negative = more anomalous)
        
        Raises:
            RuntimeError: If model is not trained
        """
        self._check_trained()
        
        logger.debug(f"Computing anomaly scores for {X.shape[0]} samples")
        return self.model.score_samples(X)
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the decision function.
        
        Args:
            X: Features (preprocessed DataFrame)
        
        Returns:
            np.ndarray: Decision function values
        
        Raises:
            RuntimeError: If model is not trained
        """
        self._check_trained()
        
        return self.model.decision_function(X)
    
    def save_model(self, filepath: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model artifact
        
        Raises:
            RuntimeError: If attempting to save untrained model
        """
        if not self.is_trained:
            raise RuntimeError(
                "Cannot save untrained model. Train the model first."
            )
        
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, filepath)
            logger.info(f"Model artifact saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving model artifact: {e}")
            raise
    
    def load_model(self, filepath: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the model artifact
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found: {filepath}")
        
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Model artifact loaded from: {filepath}")
        except Exception as e:
            logger.error(f"Error loading model artifact: {e}")
            raise
    
    def get_params(self) -> Dict:
        """
        Get model parameters.
        
        Returns:
            Dict: Model parameters
        """
        return self.model.get_params()
    
    def _check_trained(self) -> None:
        """
        Internal method to verify model is trained.
        
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before making predictions. "
                "Call train() or load_model() first."
            )