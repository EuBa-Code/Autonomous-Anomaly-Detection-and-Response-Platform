"""
monitoring.py (MONDAY 06:00) 
Read metrics_training_N.json (saved by main.py)
Compute current metrics (7-day batch)
CALL detect_drift() HERE
Save drift_report_monday.json
IF drift > 30%: trigger_training_job()
"""

from typing import Dict, Any
import logging
import json
import os
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects drift by comparing current metrics with previous ones.
    Executed ONLY from the 2nd training (Test Future) onward.
    """
    def __init__(self, contamination: float):
        self.contamination = contamination      # salvato per context e potenziale uso futuro
    def detect_drift(self, 
                     current_metrics: Dict[str, Any], 
                     previous_metrics_path: str,
                     training_number: int) -> Dict[str, Any]:
        """
        Detects drift by comparing current metrics with previous ones.

        Executed ONLY from the 2nd training (Test Future) onward.

        Args:
            current_metrics: metrics computed for the current training run (7-day batch snapshot)
            previous_metrics_path: path to metrics_training_{n-1}.json
            training_number: current training number (1, 2, 3, ...)

        
        Returns:
            dict: Structured drift report containing:
            - drift_level (float): Aggregate normalized drift score in the range [0.0, 1.0]. 0.0 = no drift, 1.0 = maximum observed drift.
            - detected (bool): Whether drift is considered actionable according to configured thresholds.
            - details (dict): Per-metric diagnostics and metadata
        
        Trigger per drift:
        - CRITICAL 🔴: |p50_prod - p50_train| > 30%
        - MODERATE 🟠: |p50_prod - p50_train| > 15%
        - MILD 🟡: |p50_prod - p50_train| > 5%
        """
        drift_report = {
            "is_first_training": training_number == 1,
            "training_number": training_number,
            "drift_detected": False,
            "drift_level": "NONE", 
            "drift_details": {}
        }
        
        # If this is the first training, do not compare
        if training_number == 1:
            logger.info("[DRIFT] First training run - no comparison possible.")
            print("\n" + "="*70)
            print("FIRST TRAINING RUN - Baseline established")
            print("="*70)
            return drift_report
        
        # load previous metrics
        try:
            if not os.path.exists(previous_metrics_path):
                logger.warning(f"[DRIFT] Previous metrics file not found: {previous_metrics_path}")
                return drift_report
            
            with open(previous_metrics_path, "r") as f:
                previous_metrics = json.load(f)
            
            logger.info(f"[DRIFT] Comparing against previous training metrics (training #{training_number - 1})")
            
        except Exception as e:
            logger.error(f"[DRIFT] Error loading previous metrics: {e}")
            return drift_report
        
        # Calculate percentage changes for selected key metrics (relative change per metric)
        print("\n" + "="*70)
        print(f"DRIFT DETECTION - Training #{training_number}")
        print("="*70)
        
        # 1. P50 Score Distribution (main metric for sensors)
        current_p50 = current_metrics["score_distribution"]["p50"]
        previous_p50 = previous_metrics["score_distribution"]["p50"]
        p50_diff_pct = abs((current_p50 - previous_p50) / abs(previous_p50)) * 100 if previous_p50 != 0 else 0
        
        # Example:
        #   current_p50 = 0.5234
        #   previous_p50 = 0.4521
        #   diff = |0.5234 - 0.4521| / 0.4521 * 100 = 15.75%
        # 
        # Interpretation: the median increased by 15.75%
        # Possible implication: data are becoming less anomalous (fewer outliers)

        drift_report["drift_details"]["p50_score"] = {
            "previous": previous_p50,
            "current": current_p50,
            "diff_pct": p50_diff_pct
        }
        
        # 2. Anomaly Rate
        current_anomaly_rate = current_metrics["anomaly_percentage"]
        previous_anomaly_rate = previous_metrics["anomaly_percentage"]
        anomaly_rate_diff_pct = abs((current_anomaly_rate - previous_anomaly_rate) / previous_anomaly_rate) * 100 if previous_anomaly_rate != 0 else 0
        
        drift_report["drift_details"]["anomaly_rate"] = {
            "previous": previous_anomaly_rate,
            "current": current_anomaly_rate,
            "diff_pct": anomaly_rate_diff_pct
        }

        # Example:
        #   current = 12.5%
        #   previous = 10.2%
        #   diff = |12.5 - 10.2| / 10.2 * 100 = 22.55%
        # 
        # Interpretation: the number of anomalies increased by 22.55%
        # Possible implication: data are deteriorating and becoming more anomalous
        
        # 3. Latency (performance degradation)
        current_latency = current_metrics["inference_latency_ms"]
        previous_latency = previous_metrics["inference_latency_ms"]
        latency_diff_pct = abs((current_latency - previous_latency) / previous_latency) * 100 if previous_latency != 0 else 0
        
        drift_report["drift_details"]["latency"] = {
            "previous": previous_latency,
            "current": current_latency,
            "diff_pct": latency_diff_pct
        }
        
        # Model performance monitoring
        # If latency increases significantly -> model is slowing down (performance degradation)

        # 4. Score mean (indicator of distributional shift)
        current_mean = current_metrics["score_statistics"]["mean"]
        previous_mean = previous_metrics["score_statistics"]["mean"]
        mean_diff_pct = abs((current_mean - previous_mean) / abs(previous_mean)) * 100 if previous_mean != 0 else 0
        
        drift_report["drift_details"]["score_mean"] = {
            "previous": previous_mean,
            "current": current_mean,
            "diff_pct": mean_diff_pct
        }

        # Has the score mean changed?
        # If the mean increased (less anomalous) or decreased (more anomalous) significantly -> possible data distribution drift

        # Determine drift severity (focus on p50 as primary metric)
        # Take the worst-case severity among the four metrics
        max_drift_pct = max(p50_diff_pct, anomaly_rate_diff_pct, mean_diff_pct)
        
        # Drift trigger based on percentage difference (thresholds are domain-dependent)
        # More than 30% change = critical condition
        # Model likely ineffective
        if max_drift_pct > 30:
            drift_report["drift_detected"] = True
            drift_report["drift_level"] = "CRITICO 🔴"
            action = " RETRAINARE IMMEDIATAMENTE!"
        
        # Between 15% and 30% = moderate situation
        elif max_drift_pct > 15:
            drift_report["drift_detected"] = True
            drift_report["drift_level"] = "MODERATO 🟠"
            action = " Monitorare attentamente - Retrainare a breve"

        # Between 5% and 15% = mild situation (small drift)
        elif max_drift_pct > 5:
            drift_report["drift_detected"] = True
            drift_report["drift_level"] = "LIEVE 🟡"
            action = " ℹ Informativo - Nessuna azione immediata"
        
        # Less than 5% = No significant drift detected (model stable)
        else:
            drift_report["drift_detected"] = False
            drift_report["drift_level"] = "NESSUNO "
            action = "Modello stabile"
        
        # Print Detailed Report
        print(f"\nDrift Level: {drift_report['drift_level']}")
        print(f"Max Drift %: {max_drift_pct:.2f}%")
        print(f"Azione: {action}")
        
        print("\nDettagli Metriche:")
        print(f"  P50 Score:      {previous_p50:.4f} → {current_p50:.4f} ({p50_diff_pct:+.2f}%)")
        print(f"  Anomaly Rate:   {previous_anomaly_rate:.2f}% → {current_anomaly_rate:.2f}% ({anomaly_rate_diff_pct:+.2f}%)")
        print(f"  Score Mean:     {previous_mean:.4f} → {current_mean:.4f} ({mean_diff_pct:+.2f}%)")
        print(f"  Latency (ms):   {previous_latency:.2f} → {current_latency:.2f} ({latency_diff_pct:+.2f}%)")
        
        print("="*70 + "\n")
        
        return drift_report