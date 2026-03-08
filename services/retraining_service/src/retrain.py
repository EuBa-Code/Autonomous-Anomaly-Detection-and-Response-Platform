"""
retrain.py — Weekly Retraining Pipeline
=========================================
Retrains the IsolationForest anomaly detector using the latest features
retrieved from the Feast feature store (Redis online store + parquet offline store).

Designed to run once per week. Airflow integration is planned for future
scheduling; for now the script can be triggered manually or via cron:
    cron example:  0 2 * * 1  →  every Monday at 02:00

Each run produces a new registered model version in MLflow so that the
inference service can always load the latest model via the MLflow Model Registry.

MLflow run records:
    Parameters  — contamination, hyperparameters, row counts, feature service name
    Metrics     — anomaly count/rate, score statistics, inference latency
    Artifacts   — thresholds.json, metrics.json
    Model       — full sklearn Pipeline (preprocessing + IsolationForest)
"""

import json
import logging
import os
import time

import mlflow
import numpy as np
from mlflow import sklearn as mlflow_sklearn

from config import Settings
from src import (
    FeatureLoader,
    ModelFactory,
    ProductionMetricsCalculator,
    create_and_log_signature,
)

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
    # FeatureLoader performs the point-in-time join between entity_df and the
    # Feast offline store, returning a clean feature DataFrame.
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
    # Integer dtype in pandas cannot represent NaN — if missing values appear at
    # inference the schema enforcement would fail with a type mismatch error.
    int_cols = x_train.select_dtypes(include="int").columns.tolist()
    if int_cols:
        x_train[int_cols] = x_train[int_cols].astype("float64")
        logger.info(f"[DATA] Cast {len(int_cols)} integer column(s) → float64: {int_cols}")

    # Cycle_Phase_ID is a categorical identifier, not a numeric magnitude.
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
        # IsolationForest converges well on a few thousand samples (Liu et al.
        # recommend max_samples=256 per tree). The subsampling cap prevents
        # RAM exhaustion when the weekly dataset grows large over time.
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
        # We run inference on the FULL dataset (not just the training subsample)
        # to compute thresholds and score distributions that reflect the entire
        # population of sensor readings — not just the 50k-row subsample.
        chunk_size = s.inference_chunk_size
        n_chunks   = (len(x_train) + chunk_size - 1) // chunk_size
        logger.info(f"[INFERENCE] {len(x_train):,} rows → {n_chunks} chunk(s) of {chunk_size:,}")

        t_infer = time.time()
        pred_parts, score_parts, pre_parts = [], [], []

        for i in range(0, len(x_train), chunk_size):
            chunk     = x_train.iloc[i : i + chunk_size]
            chunk_pre = pipe.named_steps["pre"].transform(chunk)   # preprocessing only (no re-fit)

            pred_parts.append(pipe.predict(chunk))                                          # binary labels: -1 (anomaly) / 1 (normal)
            score_parts.append(pipe.named_steps["model"].score_samples(chunk_pre))         # continuous anomaly scores
            pre_parts.append(chunk_pre)                                                     # preprocessed array (reused in evaluator)

            logger.info(f"[INFERENCE] Chunk {i // chunk_size + 1}/{n_chunks} — {len(chunk):,} rows")

        predictions = np.concatenate(pred_parts)
        scores      = np.concatenate(score_parts)
        x_train_pre = np.concatenate(pre_parts)
        del pred_parts, score_parts, pre_parts

        infer_time = time.time() - t_infer
        latency_ms = (infer_time * 1000) / len(x_train)
        logger.info(f"[INFERENCE] Completed in {infer_time:.2f}s | {latency_ms:.3f} ms/record")

        # ── 4c. EVALUATION ────────────────────────────────────────────────────
        # No ground-truth labels (unsupervised). Evaluation is based on:
        #   - Score distribution  → how well separated anomalies are from normals
        #   - Threshold extraction → p01/p05/p50 used by the inference service
        #     to classify future records without reloading the full training set
        evaluator  = ProductionMetricsCalculator(s.training.contamination)
        metrics    = evaluator.calculate_metrics(x_train_pre, predictions, scores, latency_ms, "retrain")
        thresholds = evaluator.get_thresholds(scores, predictions)

        logger.info(
            f"[EVAL] Anomalies detected: {metrics['n_anomalies_detected']} "
            f"({metrics['anomaly_percentage']:.2f}%) | "
            f"score p50: {metrics['score_distribution']['p50']:.4f}"
        )
        logger.info(
            f"[THRESHOLDS] p01={thresholds['p01']:.4f} | "
            f"p05={thresholds['p05']:.4f} | "
            f"p50={thresholds['p50']:.4f}"
        )

        # ── 4d. LOG PARAMETERS ────────────────────────────────────────────────
        mlflow.log_params({
            "feature_service":      s.feature_service_name,    # which Feast service was used
            "contamination":        s.training.contamination,
            "if_n_estimators":      s.training.if_n_estimators,
            "random_state":         s.training.random_state,
            "max_fit_rows":         max_fit_rows,
            "inference_chunk_size": chunk_size,
            "dataset_size":         len(x_train),
            "fit_rows":             min(len(x_train), max_fit_rows),
            "score_distribution":   json.dumps(metrics["score_distribution"]),
        })

        # ── 4e. LOG METRICS ───────────────────────────────────────────────────
        mlflow.log_metrics({
            "n_anomalies_detected": metrics["n_anomalies_detected"],
            "anomaly_rate":         metrics["anomaly_percentage"],
            "latency_ms":           metrics["inference_latency_ms"],
            "score_mean":           metrics["score_statistics"]["mean"],
            "score_std":            metrics["score_statistics"]["std"],
            "score_min":            metrics["score_statistics"]["min"],
            "score_max":            metrics["score_statistics"]["max"],
            "training_time_s":      round(train_time, 3),
        })

        # ── 4f. LOG MODEL ─────────────────────────────────────────────────────
        # The signature captures the raw DataFrame schema (column names + types).
        # It is inferred from the original (untransformed) DataFrame so that the
        # production inference service can send raw JSON without pre-processing.
        # Each weekly run registers a new version under the same model name,
        # making it easy to roll back to a previous week if needed.
        signature = create_and_log_signature(x_train, pipe)
        mlflow_sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            registered_model_name=s.mlflow_model_name,  # new version added each week
        )
        logger.info(f"[MLFLOW] New model version registered under '{s.mlflow_model_name}'")

        # ── 4g. LOG ARTIFACTS ─────────────────────────────────────────────────
        # thresholds.json — decision boundaries consumed by the inference service
        # metrics.json    — full evaluation snapshot for this run (drift reference)
        mlflow.log_dict(thresholds, "thresholds.json")
        mlflow.log_dict(metrics,    "metrics.json")

        # Local copy of thresholds so the inference service can read them
        # directly from the shared Docker volume without querying MLflow.
        os.makedirs(s.output_dir, exist_ok=True)
        thresholds_path = os.path.join(s.output_dir, "thresholds.json")
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        logger.info(f"[ARTIFACTS] thresholds.json saved locally to: {thresholds_path}")

    logger.info("[MAIN] Weekly retraining run completed successfully!")


if __name__ == "__main__":
    main()