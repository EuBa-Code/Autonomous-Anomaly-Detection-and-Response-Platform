"""
with mlflow.start_run():
    ├─  APRE UNA RUN (crea run_id unico)
    ├─  REGISTRA TUTTO (metriche, parametri, modello)
    └─  CHIUDE LA RUN (salva permanentemente quando esce)

##  COSA VIENE REGISTRATO:
Metriche (n_anomalies, anomaly_rate, latency, score_mean, etc)
Parametri (training_number, contamination, score_distribution)
Modello (intera pipeline con preprocessing + Isolation Forest)
Artifacts (thresholds.json, metrics.json)
Metadata (run_id, start_time, end_time, status, duration)
"""

import logging
import time
import json
import numpy as np
import os
import mlflow
from mlflow import sklearn as mlflow_sklearn
from config.settings import Settings
from src.load_from_feast import DataManager
from src.model import ModelFactory
from src.evaluator import ProductionMetricsCalculator
from src.utils import create_and_log_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_training_number(output_dir: str) -> int:
    metrics_history_file = os.path.join(output_dir, "training_history.json")
    if not os.path.exists(metrics_history_file):
        return 1
    try:
        with open(metrics_history_file, "r") as f:
            history = json.load(f)
            return len(history) + 1
    except:
        return 1

def save_training_metrics(output_dir: str, training_number: int, metrics: dict, thresholds: dict):
    metrics_history_file = os.path.join(output_dir, "training_history.json")
    metrics_by_training_file = os.path.join(output_dir, f"metrics_training_{training_number}.json")

    training_data = {
        "training_number": training_number,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "thresholds": thresholds
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(metrics_by_training_file, "w") as f:
        json.dump(training_data, f, indent=2)
    logger.info(f"[SAVE] Metriche training salvate: {metrics_by_training_file}")

    history = []
    if os.path.exists(metrics_history_file):
        try:
            with open(metrics_history_file, "r") as f:
                history = json.load(f)
        except:
            history = []

    history.append(training_data)

    with open(metrics_history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"[SAVE] Storia training aggiornata: {metrics_history_file}")

def main():
    s = Settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)

    training_number = get_training_number(s.output_dir)
    logger.info(f"[MAIN] Avvio Training #{training_number}")
    print("\n" + "="*70)
    print(f" TRAINING #{training_number}")
    print("="*70 + "\n")

    # 1. DATA
    logger.info("[DATA] Caricamento dati...")
    dm = DataManager(s)
    df = dm.load_data()

    logger.info("[DATA] Ordinamento temporale (senza split)...")
    ts = s.event_timestamp_column
    df = df.sort_values(ts).reset_index(drop=True)
    logger.info(f"[DATA] Dataset totale: {len(df)} righe ordinate temporalmente")

    drop_cols = [s.event_timestamp_column, "Machine_ID"]
    x_train = df.drop(columns=[c for c in drop_cols if c in df.columns])
    del df  # libera memoria: df originale non più necessario
    logger.info(f"[DATA] Features selezionate: {x_train.shape[1]} colonne")

    # 2. PIPELINE
    logger.info("[PIPELINE] Costruzione pipeline...")
    num_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in x_train.columns if c not in num_cols]
    logger.info(f"[PIPELINE] Colonne numeriche: {len(num_cols)} | Categoriche: {len(cat_cols)}")
    pipe = ModelFactory.build_pipeline(num_cols, cat_cols, s)

    with mlflow.start_run():
        # 3. TRAINING con subsample logico per non saturare RAM
        # Isolation Forest converge già con poche migliaia di campioni (paper: max_samples=256).
        # Il subsample è quindi teoricamente solido oltre che necessario per la RAM.
        max_fit_rows = getattr(s, "max_fit_rows", 200_000)
        if len(x_train) > max_fit_rows:
            logger.info(f"[TRAIN] Dataset grande ({len(x_train)} righe): subsample a {max_fit_rows} righe per il fit")
            x_fit = x_train.sample(n=max_fit_rows, random_state=42)
        else:
            x_fit = x_train
            logger.info(f"[TRAIN] Dataset nella norma ({len(x_train)} righe): fit su tutto il dataset")

        logger.info(f"[TRAIN] Fitting pipeline su {len(x_fit)} righe...")
        start_train = time.time()
        pipe.fit(x_fit)
        train_time = time.time() - start_train
        del x_fit  # libera il subsample di fit
        logger.info(f"[TRAIN] Training completato in {train_time:.2f}s")

        # 4. INFERENCE chunked per non saturare RAM
        logger.info("[INFERENCE] Calcolo predizioni e scores in chunk...")
        infer_chunk_size = getattr(s, "inference_chunk_size", 50_000)
        n_chunks = (len(x_train) + infer_chunk_size - 1) // infer_chunk_size
        logger.info(f"[INFERENCE] {len(x_train)} righe divise in {n_chunks} chunk da {infer_chunk_size}")

        start_inference = time.time()
        pred_parts, score_parts, pre_parts = [], [], []

        for i in range(0, len(x_train), infer_chunk_size):
            chunk = x_train.iloc[i : i + infer_chunk_size]
            chunk_pre = pipe.named_steps["pre"].transform(chunk)
            pred_parts.append(pipe.predict(chunk))
            score_parts.append(pipe.named_steps["model"].score_samples(chunk_pre))
            pre_parts.append(chunk_pre)
            chunk_idx = i // infer_chunk_size + 1
            logger.info(f"[INFERENCE] Chunk {chunk_idx}/{n_chunks} processato ({len(chunk)} righe)")

        pred_train   = np.concatenate(pred_parts)
        scores_train = np.concatenate(score_parts)
        x_train_pre  = np.concatenate(pre_parts)
        del pred_parts, score_parts, pre_parts  # libera i buffer intermedi

        inference_time = time.time() - start_inference
        latency = (inference_time * 1000) / len(x_train)
        logger.info(f"[INFERENCE] Inference completato in {inference_time:.2f}s")
        logger.info(f"[INFERENCE] Latency media: {latency:.3f}ms per record")

        # 5. EVALUATION
        logger.info("[EVAL] Calcolo metriche...")
        evaluator = ProductionMetricsCalculator(s.training.contamination)
        metrics = evaluator.calculate_metrics(x_train_pre, pred_train, scores_train, latency, "training")

        # 6. THRESHOLDS
        logger.info("[THRESHOLDS] Estrazione thresholds...")
        thresholds = evaluator.get_thresholds(scores_train, pred_train)

        logger.info(f"[EVAL] Anomalie rilevate: {metrics['n_anomalies_detected']} ({metrics['anomaly_percentage']:.2f}%)")
        logger.info(f"[EVAL] Score distribution p50: {metrics['score_distribution']['p50']:.4f}")
        logger.info(f"[THRESHOLDS] p01: {thresholds['p01']:.4f}, p05: {thresholds['p05']:.4f}, p50: {thresholds['p50']:.4f}")

        # 7. LOGGING METRICHE
        logger.info("[MLFLOW] Log metriche...")
        mlflow.log_metrics({
            "n_anomalies_detected": metrics["n_anomalies_detected"],
            "anomaly_rate": metrics["anomaly_percentage"],
            "latency_ms": metrics["inference_latency_ms"],
            "score_mean": metrics["score_statistics"]["mean"],
            "score_std": metrics["score_statistics"]["std"],
            "score_min": metrics["score_statistics"]["min"],
            "score_max": metrics["score_statistics"]["max"],
            "training_number": training_number,
            "dataset_size": len(x_train),
            "fit_rows": min(len(x_train), getattr(s, "max_fit_rows", 200_000)),
        })

        # Log statistiche features per drift monitoring
        logger.info("[MLFLOW] Log feature statistics per drift monitoring...")
        feature_stats = {}
        for col in x_train.columns:
            if x_train[col].notna().any():
                feature_stats[f"feat_{col}_mean"] = float(x_train[col].mean())
                feature_stats[f"feat_{col}_std"] = float(x_train[col].std())
                feature_stats[f"feat_{col}_p25"] = float(x_train[col].quantile(0.25))
                feature_stats[f"feat_{col}_p75"] = float(x_train[col].quantile(0.75))
        mlflow.log_metrics(feature_stats)

        # Log parametri
        mlflow.log_param("score_distribution", json.dumps(metrics["score_distribution"]))
        mlflow.log_param("training_number", str(training_number))
        mlflow.log_param("contamination", str(s.training.contamination))
        mlflow.log_param("max_fit_rows", str(getattr(s, "max_fit_rows", 200_000)))
        mlflow.log_param("inference_chunk_size", str(getattr(s, "inference_chunk_size", 50_000)))
        mlflow.log_param("feast_chunk_size", str(getattr(s, "feast_chunk_size", 50_000)))

        # Signature
        logger.info("[MLFLOW] Creazione signature...")
        signature = create_and_log_signature(x_train, pipe)

        # 8. SALVATAGGIO ARTIFACTS con log_dict (no filesystem intermedio)
        logger.info("[ARTIFACTS] Salvataggio artifacts su MLflow...")

        # Thresholds
        mlflow.log_dict(thresholds, "thresholds.json")
        logger.info("[SAVE] Thresholds salvati su MLflow: thresholds.json")

        # Metriche complete
        mlflow.log_dict(metrics, "metrics.json")
        logger.info("[SAVE] Metriche salvate su MLflow: metrics.json")

        # Salva anche localmente per compatibilità
        os.makedirs(s.output_dir, exist_ok=True)
        thresholds_file = os.path.join(s.output_dir, "thresholds.json")
        with open(thresholds_file, "w") as f:
            json.dump(thresholds, f, indent=2)

        # 9. SALVATAGGIO MODELLO
        logger.info("[MLFLOW] Log pipeline completa...")
        mlflow_sklearn.log_model(
            pipe,
            "model",
            signature=signature,
            registered_model_name=s.mlflow_model_name
        )
        logger.info(f"[MLFLOW] Pipeline salvata su MLflow (versione {training_number})")

        # Salva metriche nella storia locale
        save_training_metrics(s.output_dir, training_number, metrics, thresholds)

    logger.info(f"[MAIN] Training #{training_number} completato con successo!")
    print("\n" + "="*70)
    print(f" TRAINING #{training_number} COMPLETATO CON SUCCESSO!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()