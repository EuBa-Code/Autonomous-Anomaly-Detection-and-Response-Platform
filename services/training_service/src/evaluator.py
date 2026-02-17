import numpy as np
import logging
from typing import Dict, Any
import json
import os

logger = logging.getLogger(__name__)

class ProductionMetricsCalculator:
    """
    Cuore analitico: calcola metriche, thresholds, classifica anomalie, rileva drift.
    """
    def __init__(self, contamination: float):
        self.contamination = contamination      # salvato per contexto e potenziale uso futuro (es. alerting basato su contamination)

    def calculate_metrics(self, x_data_pre, predictions, scores, latency_ms, name="set") -> Dict[str, Any]:
        """
        Calcola metriche basandosi su predictions + scores.
        
        Args:
            x_data_pre: dati preprocessati (array)
            predictions: [-1, 1] (anomaly labels)
            scores: anomaly scores (continui, < 0 = anomalo)
            latency_ms: latenza media in millisecondi
            name: descrizione set (es "training", "test")
        
        Returns:
            dict con statistiche complete
        """
        metrics = {
            "n_anomalies_detected": int((predictions == -1).sum()), # numero di record classificati come anomalie
            "anomaly_percentage": float(((predictions == -1).sum() / len(predictions)) * 100), # percentuale di anomalie rilevate
            "score_statistics": {               # distribuzione anomaly scores
                "mean": float(np.mean(scores)), # media dei punteggi (più negativo = più anomalo)
                "std": float(np.std(scores)),   # deviazione standard dei punteggi (indica dispersione)
                "min": float(np.min(scores)),   # punteggio minimo (più anomalo)
                "max": float(np.max(scores))    # punteggio massimo (meno anomalo)
            },
            "score_distribution": {f"p{p}": float(np.percentile(scores, p)) for p in [1, 5, 50, 95, 99]},
            "inference_latency_ms": latency_ms  # latenza media per record in millisecondi
        }
        #----- Interpretazione score_distribution (percentili):-----
        # p1: 1st percentile = score del record più anomalo (che fa top 1%)
        #     Es: -2.134 → il 1% record più anomali ha score < -2.134
        # 
        # p5: 5th percentile = top 5% anomali
        #     Es: -1.567
        #
        # p50: MEDIANA = separazione tra anomali e normali
        #      Es: 0.234 → 50% record < 0.234, 50% > 0.234
        #      CRITICO per classificazione!
        #
        # p95: 95th percentile = bottom 5% (molto normale)
        #      Es: 1.892
        #
        # p99: 99th percentile = bottom 1% (massimamente normale)
        #      Es: 2.567

        return metrics

    def get_thresholds(self, scores, predictions) -> Dict[str, float]:
        """
        Estrae soglie di riferimento dai scores.
        Usate per:
        1. Classificare anomalie future
        2. Confronto drift vs training precedente
        
        Args:
            scores: anomaly scores array
            predictions: [-1, 1] labels
        
        Returns:
            dict con thresholds
        """
        return {
            "p01": float(np.percentile(scores, 1)),
            "p05": float(np.percentile(scores, 5)),
            "p50": float(np.percentile(scores, 50)), # # Separazione naturale: metà < mediana (anomali), metà > (normali)
            "observed_max_anomaly": float(np.max(scores[predictions == -1])) if any(predictions == -1) else 0.0 # se nessuna anomalia rilevata ritorna 0.0 
        }