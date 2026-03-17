"""
retrain.py — Weekly Retraining Pipeline
=========================================
Retrains the IsolationForest anomaly detector using the latest features
retrieved from the Feast feature store (Redis online store + parquet offline store).

Designed to run once per week via Airflow.

Each run produces a new registered model version in MLflow so that the
inference service can always load the latest model via the MLflow Model Registry.
"""

import json
import logging
import os
import time

import mlflow
import numpy as np
from mlflow import sklearn as mlflow_sklearn

from config.settings import Settings
from src.load_features import FeatureLoader
from src.model import ModelFactory
from src.utils import create_and_log_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Columns that identify entities but are NOT model features.
# These are dropped after feature loading so they do not pollute the training set.
_ENTITY_COLUMNS = ["Machine_ID", "created_timestamp"]


def main() -> None:
    # ── 0. CONFIGURATION ──────────────────────────────────────────────────────
    s = Settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)
    logger.info("[MAIN] Starting weekly retraining run")

    # ── 1. FEATURE LOADING (Feast) ────────────────────────────────────────────
    loader = FeatureLoader(s)
    df = loader.load()

    if df.empty:
        raise ValueError(
            "[DATA] Feature DataFrame is empty after Feast join. "
            "Check entity_df path and Feast feature service configuration."
        )

    # ── 2. FEATURE PREPARATION ────────────────────────────────────────────────
    # Drop entity/ID columns — they are not model features.
    drop_cols = [c for c in _ENTITY_COLUMNS if c in df.columns]
    if drop_cols:
        logger.info(f"[DATA] Dropping non-feature columns: {drop_cols}")

    x_train = df.drop(columns=drop_cols)
    del df  # free memory

    # Cast integer columns to float64 to safely handle missing values at inference time.
    int_cols = x_train.select_dtypes(include="int").columns.tolist()
    if int_cols:
        x_train[int_cols] = x_train[int_cols].astype("float64")
        logger.info(f"[DATA] Cast {len(int_cols)} integer column(s) → float64: {int_cols}")

    # Cycle_Phase_ID is a categorical identifier.
    if "Cycle_Phase_ID" in x_train.columns:
        x_train["Cycle_Phase_ID"] = x_train["Cycle_Phase_ID"].astype(str)
        logger.info("[DATA] Cast Cycle_Phase_ID → str (categorical)")

    logger.info(f"[DATA] Training features ({x_train.shape[1]}): {x_train.columns.tolist()}")
    logger.info(f"[DATA] Total rows available: {len(x_train):,}")

    # ── 3. PIPELINE CONSTRUCTION ──────────────────────────────────────────────
    num_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in x_train.columns if c not in num_cols]
    logger.info(f"[PIPELINE] Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

    pipe = ModelFactory.build_pipeline(num_cols, cat_cols, s)

    # ── 4. MLflow RUN ─────────────────────────────────────────────────────────
    with mlflow.start_run():

        # ── 4a. TRAINING ──────────────────────────────────────────────────────
        max_fit_rows = s.max_fit_rows
        if len(x_train) > max_fit_rows:
            logger.info(f"[TRAIN] Subsampling {len(x_train):,} → {max_fit_rows:,} rows (RAM cap)")
            x_fit = x_train.sample(n=max_fit_rows, random_state=s.training.random_state)
        else:
            x_fit = x_train
            logger.info(f"[TRAIN] Using full dataset ({len(x_train):,} rows)")

        logger.info(f"[TRAIN] Fitting pipeline on {len(x_fit):,} rows...")
        t_train = time.time()
        pipe.fit(x_fit)
        train_time = time.time() - t_train
        del x_fit
        logger.info(f"[TRAIN] Training completed in {train_time:.2f}s")

        # ── 4b. INFERENCE (chunked to limit peak RAM) ─────────────────────────
        chunk_size = s.inference_chunk_size
        n_chunks   = (len(x_train) + chunk_size - 1) // chunk_size
        logger.info(f"[INFERENCE] {len(x_train):,} rows → {n_chunks} chunk(s) of {chunk_size:,}")

        t_infer = time.time()
        pred_parts, score_parts = [], []

        for i in range(0, len(x_train), chunk_size):
            chunk     = x_train.iloc[i : i + chunk_size]
            chunk_pre = pipe.named_steps["pre"].transform(chunk)

            pred_parts.append(pipe.predict(chunk))
            score_parts.append(pipe.named_steps["model"].score_samples(chunk_pre))

            logger.info(f"[INFERENCE] Chunk {i // chunk_size + 1}/{n_chunks} — {len(chunk):,} rows")

        predictions = np.concatenate(pred_parts)
        scores      = np.concatenate(score_parts)
        del pred_parts, score_parts

        infer_time = time.time() - t_infer
        latency_ms = (infer_time * 1000) / len(x_train)
        logger.info(f"[INFERENCE] Completed in {infer_time:.2f}s | {latency_ms:.3f} ms/record")

        # ── 4c. COMPUTE METRICS ───────────────────────────────────────────────
        anomaly_count = int((predictions == -1).sum())
        normal_count = int((predictions == 1).sum())
        anomaly_rate = anomaly_count / len(predictions)
        score_percentiles = np.percentile(scores, [1, 5, 10, 25, 50, 75, 90, 95, 99])

        logger.info(f"[METRICS] Anomalies: {anomaly_count:,} ({anomaly_rate:.2%})")
        logger.info(f"[METRICS] Normal: {normal_count:,} ({1 - anomaly_rate:.2%})")
        logger.info(f"[METRICS] Score mean: {scores.mean():.4f} | std: {scores.std():.4f}")

        # ── 4d. LOG PARAMETERS ────────────────────────────────────────────────
        mlflow.log_params({
            "feature_service":      s.feature_service_name,
            "contamination":        s.training.contamination,
            "if_n_estimators":      s.training.if_n_estimators,
            "random_state":         s.training.random_state,
            "max_fit_rows":         max_fit_rows,
            "inference_chunk_size": chunk_size,
            "dataset_size":         len(x_train),
            "fit_rows":             min(len(x_train), max_fit_rows),
            "n_numeric_features":   len(num_cols),
            "n_categorical_features": len(cat_cols),
        })

        # ── 4e. LOG METRICS ───────────────────────────────────────────────────
        mlflow.log_metrics({
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_rate": anomaly_rate,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_p01": float(score_percentiles[0]),
            "score_p05": float(score_percentiles[1]),
            "score_p10": float(score_percentiles[2]),
            "score_p25": float(score_percentiles[3]),
            "score_p50": float(score_percentiles[4]),
            "score_p75": float(score_percentiles[5]),
            "score_p90": float(score_percentiles[6]),
            "score_p95": float(score_percentiles[7]),
            "score_p99": float(score_percentiles[8]),
            "train_time_sec": train_time,
            "inference_time_sec": infer_time,
            "latency_ms_per_record": latency_ms,
        })

        # ── 4f. EXPORT METRICS JSON ───────────────────────────────────────────
        os.makedirs(s.output_dir, exist_ok=True)
        
        metrics_export = {
            "dataset": {
                "total_rows": len(x_train),
                "fit_rows": min(len(x_train), max_fit_rows),
                "n_features": x_train.shape[1],
                "n_numeric_features": len(num_cols),
                "n_categorical_features": len(cat_cols),
            },
            "anomaly_detection": {
                "anomaly_count": anomaly_count,
                "normal_count": normal_count,
                "anomaly_rate": float(anomaly_rate),
            },
            "score_distribution": {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "percentiles": {
                    "p01": float(score_percentiles[0]),
                    "p05": float(score_percentiles[1]),
                    "p10": float(score_percentiles[2]),
                    "p25": float(score_percentiles[3]),
                    "p50": float(score_percentiles[4]),
                    "p75": float(score_percentiles[5]),
                    "p90": float(score_percentiles[6]),
                    "p95": float(score_percentiles[7]),
                    "p99": float(score_percentiles[8]),
                },
            },
            "performance": {
                "train_time_sec": train_time,
                "inference_time_sec": infer_time,
                "latency_ms_per_record": latency_ms,
            },
        }

        metrics_path = os.path.join(s.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_export, f, indent=2)
        
        mlflow.log_artifact(metrics_path)
        logger.info(f"[ARTIFACTS] Exported metrics to {metrics_path}")

        # ── 4g. LOG MODEL ─────────────────────────────────────────────────────
        signature = create_and_log_signature(x_train, pipe)
        mlflow_sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            registered_model_name=s.mlflow_model_name,
        )
        logger.info(f"[MLFLOW] New model version registered under '{s.mlflow_model_name}'")

    logger.info("[MAIN] Weekly retraining run completed successfully!")


if __name__ == "__main__":
    main()