import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ProductionMetricsCalculator:
    """
    # Analytical core: computes metrics, determines thresholds, classifies anomalies, and detects drift.
    """
    def __init__(self, contamination: float):
        self.contamination = contamination      # Saved for context and potential future use (e.g., alerting based on contamination)

    def calculate_metrics(self, x_data_pre, predictions, scores, latency_ms, name="set") -> Dict[str, Any]:
        """
        Compute metrics based on predictions + scores.

        Args:
            x_data_pre: preprocessed data (array)
            predictions: [-1, 1] (anomaly labels)
            scores: anomaly scores (continuous, < 0 = anomalous)
            latency_ms: average latency in milliseconds
            name: dataset description (e.g., "training", "test")

        Returns:
            dict with complete statistics

        """
        metrics = {
            "n_anomalies_detected": int((predictions == -1).sum()), # number of records classified as anomalies
            "anomaly_percentage": float(((predictions == -1).sum() / len(predictions)) * 100), # percentage of detected anomalies
            "score_statistics": {               # anomaly scores distribution
                "mean": float(np.mean(scores)), # scores mean (more negative = more anomalous)
                "std": float(np.std(scores)),   # scores standard deviation (dispersion indicator)
                "min": float(np.min(scores)),   # minimum score (most anomalous)
                "max": float(np.max(scores))    # maximum score (least anomalous)
            },
            "score_distribution": {f"p{p}": float(np.percentile(scores, p)) for p in [1, 5, 50, 95, 99]},
            "inference_latency_ms": latency_ms  # avg latency per record (ms)
        }
        #----- Score_distribution Interpretation (%):-----
        # p1: 1st percentile = score of the most anomalous record (top 1%)
        #     Es: -2.134 → the top 1% most anomalous records have score < -2.134
        # 
        # p5: 5th percentile = top 5% anomalous
        #     Es: -1.567
        #
        # p50: MEDIAN = separation between anomalous and normal
        #      Es: 0.234 → 50% of records < 0.234, 50% > 0.234
        #      CRITICAL for classification!
        #
        # p95: 95th percentile = bottom 5% (- low anomaly score)
        #      Es: 1.892
        #
        # p99: 99th percentile = bottom 1% (- minimum anomaly score)
        #      Es: 2.567

        return metrics

    def get_thresholds(self, scores, predictions) -> Dict[str, float]:
        """
        Extracts reference thresholds from the scores.
        Used for:
            1. Classifying future anomalies
            2. Comparing drift against previous training

        Args:
            scores: array of anomaly scores
            predictions: labels [-1, 1]

        Returns:
            dict with thresholds

        """
        return {
            "p01": float(np.percentile(scores, 1)),
            "p05": float(np.percentile(scores, 5)),
            "p50": float(np.percentile(scores, 50)), # Natural separation: half < median (anomalous), half > median (normal)
            "observed_max_anomaly": float(np.max(scores[predictions == -1])) if any(predictions == -1) else 0.0 # returns 0.0 if no anomalies detected
        }