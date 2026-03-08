"""
Training pipeline — Datalake Only
===================================
Loads data exclusively from the datalake (parquet files).
No Feast / feature-store integration.

MLflow run records:
    Metrics    — anomaly counts, rates, latency, score statistics,
                 per-feature drift statistics
    Parameters — contamination, training number, row counts
    Artifacts  — thresholds.json, metrics.json
    Model      — full sklearn Pipeline (preprocessing + IsolationForest)
"""

import json
import logging
import os
import time

import mlflow
import numpy as np
from mlflow import sklearn as mlflow_sklearn

from config.settings import Settings
from src.load_from_datalake import DataManager
from src.model import ModelFactory
from src.evaluator import ProductionMetricsCalculator
from src.utils import create_and_log_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_training_number(output_dir: str) -> int:
    """Return the next training number by counting entries in training_history.json."""
    history_file = os.path.join(output_dir, "training_history.json")
    if not os.path.exists(history_file):
        return 1
    try:
        with open(history_file) as f:
            return len(json.load(f)) + 1
    except Exception:
        return 1


def save_training_metrics(
    output_dir: str,
    training_number: int,
    metrics: dict,
    thresholds: dict,
) -> None:
    """Persist per-run metrics and append to the cumulative training history."""
    os.makedirs(output_dir, exist_ok=True)

    run_data = {
        "training_number": training_number,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "thresholds": thresholds,
    }

    # Per-run file
    run_file = os.path.join(output_dir, f"metrics_training_{training_number}.json")
    with open(run_file, "w") as f:
        json.dump(run_data, f, indent=2)
    logger.info(f"[SAVE] Run metrics saved: {run_file}")

    # Cumulative history
    history_file = os.path.join(output_dir, "training_history.json")
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(run_data)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"[SAVE] Training history updated: {history_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    s = Settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)

    training_number = get_training_number(s.output_dir)

    print("\n" + "=" * 70)
    print(f"  TRAINING #{training_number}  [Datalake Only]")
    print("=" * 70 + "\n")
    logger.info(f"[MAIN] Starting Training #{training_number}")

    # ── 1. DATA ───────────────────────────────────────────────────────────────
    logger.info("[DATA] Loading data from datalake...")
    dm = DataManager(s)
    df = dm.load_data()

    if df.empty:
        raise ValueError(
            f"[DATA] Dataset is empty. "
            f"Check datalake path: {s.entity_df_path}"
        )

    ts_col = s.event_timestamp_column
    if ts_col not in df.columns:
        raise KeyError(
            f"[DATA] Timestamp column '{ts_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    logger.info("[DATA] Sorting by timestamp (no train/test split)...")
    df = df.sort_values(ts_col).reset_index(drop=True)
    logger.info(f"[DATA] Total rows (chronological order): {len(df):,}")

    # Drop timestamp and common ID columns — they are not model features
    drop_cols = [ts_col]
    for id_col in ("Machine_ID", "machine_id", "entity_id", "id"):
        if id_col in df.columns:
            drop_cols.append(id_col)

    x_train = df.drop(columns=[c for c in drop_cols if c in df.columns])
    del df  # free original DataFrame
    logger.info(f"[DATA] Feature columns ({x_train.shape[1]}): {x_train.columns.tolist()}")

    # ── 2. PIPELINE ───────────────────────────────────────────────────────────
    num_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in x_train.columns if c not in num_cols]
    logger.info(
        f"[PIPELINE] Numeric: {len(num_cols)} cols | Categorical: {len(cat_cols)} cols"
    )
    pipe = ModelFactory.build_pipeline(num_cols, cat_cols, s)

    with mlflow.start_run():

        # ── 3. TRAINING ───────────────────────────────────────────────────────
        # Subsample to avoid RAM exhaustion; IsolationForest converges well
        # with a few thousand samples (paper recommends max_samples=256).
        max_fit_rows = s.max_fit_rows
        if len(x_train) > max_fit_rows:
            logger.info(
                f"[TRAIN] Subsampling {len(x_train):,} → {max_fit_rows:,} rows "
                f"(memory-aware cap)"
            )
            x_fit = x_train.sample(n=max_fit_rows, random_state=42)
        else:
            x_fit = x_train
            logger.info(f"[TRAIN] Training on full dataset ({len(x_train):,} rows)")

        logger.info(f"[TRAIN] Fitting pipeline on {len(x_fit):,} rows...")
        t_train = time.time()
        pipe.fit(x_fit)
        train_time = time.time() - t_train
        del x_fit
        logger.info(f"[TRAIN] Training completed in {train_time:.2f}s")

        # ── 4. INFERENCE (chunked) ────────────────────────────────────────────
        chunk_size = s.inference_chunk_size
        n_chunks = (len(x_train) + chunk_size - 1) // chunk_size
        logger.info(
            f"[INFERENCE] {len(x_train):,} rows → {n_chunks} chunk(s) of {chunk_size:,}"
        )

        t_infer = time.time()
        pred_parts, score_parts, pre_parts = [], [], []

        for i in range(0, len(x_train), chunk_size):
            chunk = x_train.iloc[i : i + chunk_size]
            chunk_pre = pipe.named_steps["pre"].transform(chunk)
            pred_parts.append(pipe.predict(chunk))
            score_parts.append(pipe.named_steps["model"].score_samples(chunk_pre))
            pre_parts.append(chunk_pre)
            idx = i // chunk_size + 1
            logger.info(f"[INFERENCE] Chunk {idx}/{n_chunks} — {len(chunk):,} rows")

        predictions  = np.concatenate(pred_parts)
        scores       = np.concatenate(score_parts)
        x_train_pre  = np.concatenate(pre_parts)
        del pred_parts, score_parts, pre_parts

        infer_time = time.time() - t_infer
        latency_ms = (infer_time * 1000) / len(x_train)
        logger.info(
            f"[INFERENCE] Completed in {infer_time:.2f}s "
            f"| mean latency: {latency_ms:.3f} ms/record"
        )

        # ── 5. EVALUATION ─────────────────────────────────────────────────────
        evaluator = ProductionMetricsCalculator(s.training.contamination)
        metrics   = evaluator.calculate_metrics(
            x_train_pre, predictions, scores, latency_ms, "training"
        )
        thresholds = evaluator.get_thresholds(scores, predictions)

        logger.info(
            f"[EVAL] Anomalies: {metrics['n_anomalies_detected']} "
            f"({metrics['anomaly_percentage']:.2f}%) | "
            f"score p50: {metrics['score_distribution']['p50']:.4f}"
        )
        logger.info(
            f"[THRESHOLDS] p01={thresholds['p01']:.4f}  "
            f"p05={thresholds['p05']:.4f}  p50={thresholds['p50']:.4f}"
        )

        # ── 6. MLFLOW — METRICS ───────────────────────────────────────────────
        mlflow.log_metrics({
            "n_anomalies_detected": metrics["n_anomalies_detected"],
            "anomaly_rate":         metrics["anomaly_percentage"],
            "latency_ms":           metrics["inference_latency_ms"],
            "score_mean":           metrics["score_statistics"]["mean"],
            "score_std":            metrics["score_statistics"]["std"],
            "score_min":            metrics["score_statistics"]["min"],
            "score_max":            metrics["score_statistics"]["max"],
            "training_number":      training_number,
            "dataset_size":         len(x_train),
            "fit_rows":             min(len(x_train), max_fit_rows),
        })

        # Per-feature drift statistics
        feature_stats = {}
        for col in x_train.columns:
            if x_train[col].notna().any():
                feature_stats[f"feat_{col}_mean"] = float(x_train[col].mean())
                feature_stats[f"feat_{col}_std"]  = float(x_train[col].std())
                feature_stats[f"feat_{col}_p25"]  = float(x_train[col].quantile(0.25))
                feature_stats[f"feat_{col}_p75"]  = float(x_train[col].quantile(0.75))
        mlflow.log_metrics(feature_stats)

        # ── 7. MLFLOW — PARAMETERS ────────────────────────────────────────────
        mlflow.log_param("score_distribution",    json.dumps(metrics["score_distribution"]))
        mlflow.log_param("training_number",       str(training_number))
        mlflow.log_param("contamination",         str(s.training.contamination))
        mlflow.log_param("max_fit_rows",          str(max_fit_rows))
        mlflow.log_param("inference_chunk_size",  str(chunk_size))
        mlflow.log_param("data_source",           "datalake_only")

        # ── 8. MLFLOW — SIGNATURE + MODEL ─────────────────────────────────────
        signature = create_and_log_signature(x_train, pipe)
        mlflow_sklearn.log_model(
            pipe,
            "model",
            signature=signature,
            registered_model_name=s.mlflow_model_name,
        )
        logger.info(f"[MLFLOW] Pipeline logged (training #{training_number})")

        # ── 9. MLFLOW — ARTIFACTS ─────────────────────────────────────────────
        mlflow.log_dict(thresholds, "thresholds.json")
        mlflow.log_dict(metrics,    "metrics.json")

        # Also write thresholds locally for downstream services
        os.makedirs(s.output_dir, exist_ok=True)
        with open(os.path.join(s.output_dir, "thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=2)

        # ── 10. LOCAL HISTORY ─────────────────────────────────────────────────
        save_training_metrics(s.output_dir, training_number, metrics, thresholds)

    logger.info(f"[MAIN] Training #{training_number} completed successfully!")
    print("\n" + "=" * 70)
    print(f"  TRAINING #{training_number} COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()