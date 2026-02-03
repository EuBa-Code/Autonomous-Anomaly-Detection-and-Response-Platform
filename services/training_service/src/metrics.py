"""
Metrics calculation module for model evaluation.
"""
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import json

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate and manage evaluation metrics for anomaly detection.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics: Dict = {}
        logger.debug("MetricsCalculator initialized")

    def calculate_supervised_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate supervised metrics when ground truth labels are available.
        
        Args:
            y_true: True labels (0 = normal, 1 = anomaly)
            y_pred: Predicted labels (-1 = anomaly, 1 = normal from IsolationForest)
            scores: Anomaly scores (optional, for AUC calculation)
        
        Returns:
            Dict: Calculated metrics
        """
        # Convert IsolationForest predictions (-1, 1) to (1, 0)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            
            # Additional metrics
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # AUC if scores are provided
            if scores is not None:
                # For IsolationForest, lower scores = more anomalous
                # So we negate scores for AUC calculation
                metrics['roc_auc'] = roc_auc_score(y_true, -scores)
            
            # Anomaly statistics
            metrics['total_samples'] = len(y_true)
            metrics['total_anomalies_true'] = int(y_true.sum())
            metrics['total_anomalies_predicted'] = int(y_pred_binary.sum())
            metrics['anomaly_rate_true'] = float(y_true.mean())
            metrics['anomaly_rate_predicted'] = float(y_pred_binary.mean())
            
            logger.info("Supervised metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating supervised metrics: {e}")
            raise
        
        self.metrics = metrics
        return metrics

    def calculate_unsupervised_metrics(
        self,
        y_pred: np.ndarray,
        scores: np.ndarray
    ) -> Dict:
        """
        Calculate unsupervised metrics when no ground truth is available.
        
        Args:
            y_pred: Predicted labels (-1 = anomaly, 1 = normal)
            scores: Anomaly scores
        
        Returns:
            Dict: Calculated metrics
        """
        # Convert predictions to binary
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        metrics = {}
        
        try:
            # Basic statistics
            metrics['total_samples'] = len(y_pred)
            metrics['total_anomalies_predicted'] = int(y_pred_binary.sum())
            metrics['anomaly_rate'] = float(y_pred_binary.mean())
            
            # Score statistics
            metrics['score_mean'] = float(scores.mean())
            metrics['score_std'] = float(scores.std())
            metrics['score_min'] = float(scores.min())
            metrics['score_max'] = float(scores.max())
            metrics['score_median'] = float(np.median(scores))
            
            # Percentiles
            metrics['score_p5'] = float(np.percentile(scores, 5))
            metrics['score_p25'] = float(np.percentile(scores, 25))
            metrics['score_p75'] = float(np.percentile(scores, 75))
            metrics['score_p95'] = float(np.percentile(scores, 95))
            
            logger.info("Unsupervised metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating unsupervised metrics: {e}")
            raise
        
        self.metrics = metrics
        return metrics

    def save_metrics(self, filepath: Path) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            filepath: Path where to save metrics
        """
        if not self.metrics:
            logger.warning("No metrics to save")
            return
        
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            logger.info(f"Metrics saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise

    def print_summary(self) -> None:
        """Print a formatted summary of metrics."""
        if not self.metrics:
            logger.warning("No metrics available to print")
            return
        
        logger.info("=" * 60)
        logger.info("METRICS SUMMARY")
        logger.info("=" * 60)
        
        for key, value in self.metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:30s}: {value:.4f}")
            else:
                logger.info(f"{key:30s}: {value}")
        
        logger.info("=" * 60)