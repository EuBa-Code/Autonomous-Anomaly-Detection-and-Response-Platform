"""
train.py — First Training Run
==============================
Simplified pipeline for an initial IsolationForest training.
Loads data from the datalake (parquet), fits the model, evaluates it,
and logs everything to MLflow.

MLflow run records:
    Parameters  — contamination, row counts, subsampling cap, chunk size
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

from config.settings import Settings
from src.evaluator import ProductionMetricsCalculator
from src.load_from_datalake import DataManager
from src.model import ModelFactory
from src.utils import create_and_log_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # ── 0. CONFIGURATION ──────────────────────────────────────────────────────
    s = Settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)
    logger.info("[MAIN] Starting first training run")

    # ── 1. DATA LOADING ───────────────────────────────────────────────────────
    logger.info("[DATA] Loading data from datalake...")
    dm = DataManager(s)
    df = dm.load_data()

    if df.empty:
        raise ValueError(
            f"[DATA] Dataset is empty. Check datalake path: {s.entity_df_path}"
        )

    ts_col = s.event_timestamp_column
    if ts_col not in df.columns:
        raise KeyError(
            f"[DATA] Timestamp column '{ts_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Sort chronologically before dropping the timestamp
    logger.info("[DATA] Sorting by timestamp...")
    df = df.sort_values(ts_col).reset_index(drop=True)
    logger.info(f"[DATA] Total rows: {len(df):,}")

    # Drop non-feature columns (IDs and timestamps are not model inputs)
    drop_cols = [c for c in ["Machine_ID", "timestamp"] if c in df.columns]
    x_train = df.drop(columns=drop_cols)
    del df  # free memory
    logger.info(f"[DATA] Feature columns ({x_train.shape[1]}): {x_train.columns.tolist()}")

    # ── 2. PIPELINE CONSTRUCTION ──────────────────────────────────────────────
    num_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in x_train.columns if c not in num_cols]
    logger.info(f"[PIPELINE] Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    pipe = ModelFactory.build_pipeline(num_cols, cat_cols, s)

    # ── 3. MLflow RUN ─────────────────────────────────────────────────────────
    with mlflow.start_run():

        # ── 3a. TRAINING ──────────────────────────────────────────────────────
        # IsolationForest converges well on a few thousand samples;
        # subsampling avoids RAM exhaustion on large datasets.
        max_fit_rows = s.max_fit_rows
        if len(x_train) > max_fit_rows:
            logger.info(f"[TRAIN] Subsampling {len(x_train):,} → {max_fit_rows:,} rows")
            x_fit = x_train.sample(n=max_fit_rows, random_state=42)
        else:
            x_fit = x_train
            logger.info(f"[TRAIN] Training on full dataset ({len(x_train):,} rows)")

        logger.info(f"[TRAIN] Fitting pipeline on {len(x_fit):,} rows...")
        t_train = time.time()
        pipe.fit(x_fit)
        train_time = time.time() - t_train
        del x_fit
        logger.info(f"[TRAIN] Completed in {train_time:.2f}s")

        # ── 3b. INFERENCE (chunked to limit peak RAM) ─────────────────────────
        chunk_size = s.inference_chunk_size
        n_chunks = (len(x_train) + chunk_size - 1) // chunk_size
        logger.info(f"[INFERENCE] {len(x_train):,} rows → {n_chunks} chunk(s) of {chunk_size:,}")

        t_infer = time.time()
        pred_parts, score_parts, pre_parts = [], [], []

        for i in range(0, len(x_train), chunk_size):
            chunk     = x_train.iloc[i : i + chunk_size]
            chunk_pre = pipe.named_steps["pre"].transform(chunk)

            pred_parts.append(pipe.predict(chunk))
            score_parts.append(pipe.named_steps["model"].score_samples(chunk_pre))
            pre_parts.append(chunk_pre)

            logger.info(f"[INFERENCE] Chunk {i // chunk_size + 1}/{n_chunks} — {len(chunk):,} rows")

        predictions = np.concatenate(pred_parts)
        scores      = np.concatenate(score_parts)
        x_train_pre = np.concatenate(pre_parts)
        del pred_parts, score_parts, pre_parts

        infer_time = time.time() - t_infer
        latency_ms = (infer_time * 1000) / len(x_train)
        logger.info(f"[INFERENCE] Completed in {infer_time:.2f}s | {latency_ms:.3f} ms/record")

        # ── 3c. EVALUATION ────────────────────────────────────────────────────
        evaluator  = ProductionMetricsCalculator(s.training.contamination)
        metrics    = evaluator.calculate_metrics(x_train_pre, predictions, scores, latency_ms, "training")
        thresholds = evaluator.get_thresholds(scores, predictions)

        logger.info(
            f"[EVAL] Anomalies: {metrics['n_anomalies_detected']} "
            f"({metrics['anomaly_percentage']:.2f}%) | "
            f"score p50: {metrics['score_distribution']['p50']:.4f}"
        )
        logger.info(
            f"[THRESHOLDS] p01={thresholds['p01']:.4f} | "
            f"p05={thresholds['p05']:.4f} | p50={thresholds['p50']:.4f}"
        )

        # ── 3d. LOG PARAMETERS ────────────────────────────────────────────────
        mlflow.log_params({
            "contamination":        s.training.contamination,
            "max_fit_rows":         max_fit_rows,
            "inference_chunk_size": chunk_size,
            "dataset_size":         len(x_train),
            "fit_rows":             min(len(x_train), max_fit_rows),
            "score_distribution":   json.dumps(metrics["score_distribution"]),
        })

        # ── 3e. LOG METRICS ───────────────────────────────────────────────────
        mlflow.log_metrics({
            "n_anomalies_detected": metrics["n_anomalies_detected"],
            "anomaly_rate":         metrics["anomaly_percentage"],
            "latency_ms":           metrics["inference_latency_ms"],
            "score_mean":           metrics["score_statistics"]["mean"],
            "score_std":            metrics["score_statistics"]["std"],
            "score_min":            metrics["score_statistics"]["min"],
            "score_max":            metrics["score_statistics"]["max"],
        })

        # ── 3f. LOG MODEL ─────────────────────────────────────────────────────
        # Signature maps raw DataFrame inputs → model output labels.
        # Always infer from the original (untransformed) DataFrame so that
        # the production API can send JSON directly without pre-processing.
        signature = create_and_log_signature(x_train, pipe)
        mlflow_sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            registered_model_name=s.mlflow_model_name,
        )
        logger.info("[MLFLOW] Pipeline registered successfully")

        # ── 3g. LOG ARTIFACTS ─────────────────────────────────────────────────
        # thresholds.json — decision boundaries for future inference services
        # metrics.json    — full evaluation snapshot for this training run
        mlflow.log_dict(thresholds, "thresholds.json")
        mlflow.log_dict(metrics,    "metrics.json")

        # Write thresholds locally so downstream services can read them
        # without hitting MLflow (optional convenience copy)
        os.makedirs(s.output_dir, exist_ok=True)
        thresholds_path = os.path.join(s.output_dir, "thresholds.json")
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        logger.info(f"[ARTIFACTS] thresholds.json saved to {thresholds_path}")

    logger.info("[MAIN] First training run completed successfully!")


if __name__ == "__main__":
    main()