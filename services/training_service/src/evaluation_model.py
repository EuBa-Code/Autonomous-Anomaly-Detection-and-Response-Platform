"""
Module for calculating evaluation metrics.
"""
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    silhouette_score
)

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Class to calculate and manage model metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, 
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          scores: Optional[np.ndarray] = None,
                          X: Optional[np.ndarray] = None) -> Dict:
        """
        Calculates all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            scores: Anomaly scores
            X: Feature matrix (optional, needed for silhouette score)
        
        Returns:
            Dict: Dictionary with all metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        # Convert predictions from {-1, 1} to {1, 0} for compatibility
        # -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        # If y_true has {-1, 1}, convert it
        if set(np.unique(y_true)) == {-1, 1}:
            y_true_binary = np.where(y_true == -1, 1, 0)
        else:
            y_true_binary = y_true
        
        # Primary metrics for anomaly detection (priority order)
        metrics = {
            'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            'f1_score': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            'roc_auc': None,
            'silhouette_score': None
        }
        
        # ROC AUC if scores are available (important for ranking)
        if scores is not None:
            try:
                # Invert scores (more negative = more anomalous)
                metrics['roc_auc'] = float(roc_auc_score(y_true_binary, -scores))
            except Exception as e:
                logger.warning(f"Unable to calculate ROC AUC: {e}")
                metrics['roc_auc'] = None

        # Silhouette Score if features X are provided (clustering metric)
        if X is not None and len(np.unique(y_pred)) > 1:
            try:
                sil_score = silhouette_score(X, y_pred)
                metrics['silhouette_score'] = float(sil_score)
                logger.info(f"Silhouette Score: {sil_score:.4f}")
            except Exception as e:
                logger.warning(f"Unable to calculate Silhouette Score: {e}")
                metrics['silhouette_score'] = None
        elif X is None:
            logger.info("Features X not provided, Silhouette Score not calculated")
        elif len(np.unique(y_pred)) <= 1:
            logger.warning("All predictions belong to the same class, Silhouette Score cannot be calculated")

        # Accuracy (secondary metric, can be misleading with imbalanced data)
        metrics['accuracy'] = float(accuracy_score(y_true_binary, y_pred_binary))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        # Predictions statistics
        metrics['predictions_stats'] = {
            'total_samples': int(len(y_pred)),
            'predicted_anomalies': int(np.sum(y_pred_binary)),
            'predicted_normal': int(np.sum(1 - y_pred_binary)),
            'anomaly_rate': float(np.mean(y_pred_binary))
        }
        
        # True labels stats if available
        metrics['true_labels_stats'] = {
            'total_samples': int(len(y_true_binary)),
            'true_anomalies': int(np.sum(y_true_binary)),
            'true_normal': int(np.sum(1 - y_true_binary)),
            'true_anomaly_rate': float(np.mean(y_true_binary))
        }
        
        # Classification report
        try:
            report = classification_report(
                y_true_binary, 
                y_pred_binary,
                target_names=['Normal', 'Anomaly'],
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = report
        except Exception as e:
            logger.warning(f"Unable to generate classification report: {e}")
        
        self.metrics = metrics
        self._log_metrics()
        
        return metrics
    
    def calculate_unsupervised_metrics(self,
                                       y_pred: np.ndarray,
                                       scores: np.ndarray,
                                       X: Optional[np.ndarray] = None) -> Dict:
        """
        Calculates metrics for unsupervised case (without true labels)
        
        Args:
            y_pred: Model predictions
            scores: Anomaly scores
            X: Feature matrix (optional, needed for silhouette score)
        Returns:
            Dict: Dictionary with unsupervised metrics
        """
        logger.info("Calculating unsupervised metrics...")
        
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        metrics = {
            'predictions_stats': {
                'total_samples': int(len(y_pred)),
                'predicted_anomalies': int(np.sum(y_pred_binary)),
                'predicted_normal': int(np.sum(1 - y_pred_binary)),
                'anomaly_rate': float(np.mean(y_pred_binary))
            },
            'score_statistics': {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'median_score': float(np.median(scores)),
                'q25_score': float(np.percentile(scores, 25)),
                'q75_score': float(np.percentile(scores, 75))
            },
            'silhouette_score': None
        }

        # Silhouette Score if features X are provided
        if X is not None and len(np.unique(y_pred)) > 1:
            try:
                sil_score = silhouette_score(X, y_pred)
                metrics['silhouette_score'] = float(sil_score)
                logger.info(f"Silhouette Score: {sil_score:.4f}")
            except Exception as e:
                logger.warning(f"Unable to calculate Silhouette Score: {e}")
                metrics['silhouette_score'] = None
        elif X is None:
            logger.info("Features X not provided, Silhouette Score not calculated")
        elif len(np.unique(y_pred)) <= 1:
            logger.warning("All predictions belong to the same class, Silhouette Score cannot be calculated")
        
        self.metrics = metrics
        self._log_metrics()
        
        return metrics
    
    def _log_metrics(self):
        """Logs the main metrics emphasizing those most relevant for anomaly detection"""
        if 'precision' in self.metrics:
            logger.info("=" * 60)
            logger.info("PRIMARY METRICS (Anomaly Detection)")
            logger.info("=" * 60)
            logger.info(f"Precision:  {self.metrics['precision']:.4f}  (Correct predicted anomalies)")
            logger.info(f"Recall:     {self.metrics['recall']:.4f}  (True anomalies identified)")
            logger.info(f"F1-Score:   {self.metrics['f1_score']:.4f}  (Balance between precision/recall)")
            
            if self.metrics.get('roc_auc'):
                logger.info(f"ROC AUC:    {self.metrics['roc_auc']:.4f}  (Ranking capability)")

            if self.metrics.get('silhouette_score') is not None:
                logger.info(f"Silhouette: {self.metrics['silhouette_score']:.4f}  (Clustering quality)")
            
            logger.info("-" * 60)
            logger.info("SECONDARY METRICS")
            logger.info("-" * 60)
            logger.info(f"Accuracy:   {self.metrics['accuracy']:.4f}  (Can be misleading with imbalanced data)")
            logger.info("=" * 60)
        
        if 'predictions_stats' in self.metrics:
            stats = self.metrics['predictions_stats']
            logger.info(f"\nPredicted anomalies: {stats['predicted_anomalies']}/{stats['total_samples']} "
                        f"({stats['anomaly_rate']:.2%})")
            
        if 'true_labels_stats' in self.metrics:
            stats = self.metrics['true_labels_stats']
            logger.info(f"True anomalies:      {stats['true_anomalies']}/{stats['total_samples']} "
                        f"({stats['true_anomaly_rate']:.2%})")
    
    def save_metrics(self, filepath: Path):
        """
        Saves metrics to a JSON file
        
        Args:
            filepath: Path where to save metrics
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise
    
    def load_metrics(self, filepath: Path) -> Dict:
        """
        Loads metrics from a JSON file
        
        Args:
            filepath: Path of the metrics file
        
        Returns:
            Dict: Loaded metrics
        """
        try:
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"Metrics loaded from {filepath}")
            return self.metrics
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            raise
    
    def get_metrics(self) -> Dict:
        """Returns the calculated metrics"""
        return self.metrics
    
    def print_summary(self):
        """Prints a formatted summary of the main metrics"""
        if not self.metrics:
            logger.warning("No metrics available")
            return
        
        print("\n" + "=" * 70)
        print(" METRICS SUMMARY - ANOMALY DETECTION")
        print("=" * 70)
        
        if 'precision' in self.metrics:
            print(f"\nPrimary Metrics")
            print(f"  Precision:     {self.metrics['precision']:6.2%}  | Quality of predicted anomalies")
            print(f"  Recall:        {self.metrics['recall']:6.2%}  | Coverage of true anomalies")
            print(f"  F1-Score:      {self.metrics['f1_score']:6.2%}  | Harmonic mean P/R")
            if self.metrics.get('roc_auc'):
                print(f"  ROC AUC:       {self.metrics['roc_auc']:6.2%}  | Discriminative ability")
            if self.metrics.get('silhouette_score') is not None:
                print(f"  Silhouette:    {self.metrics['silhouette_score']:6.4f}  | Clustering quality [-1, 1]")
            
            print(f"\nSecondary Metrics")
            print(f"  Accuracy:      {self.metrics['accuracy']:6.2%}  | (Warning: imbalanced data)")
        
        if 'confusion_matrix' in self.metrics:
            cm = self.metrics['confusion_matrix']
            print(f"\nConfusion Matrix")
            print(f"                   Predicted Normal  |  Predicted Anomaly")
            print(f"  True Normal      {cm['true_negative']:6d}           |  {cm['false_positive']:6d}  (FP)")
            print(f"  True Anomaly     {cm['false_negative']:6d}  (FN)     |  {cm['true_positive']:6d}  (TP)")
        
        print("\n" + "=" * 70 + "\n")