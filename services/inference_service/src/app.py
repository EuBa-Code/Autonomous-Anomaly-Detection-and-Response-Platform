"""
Inference Pipeline — Washing Machine Anomaly Detection
======================================================

  - Model loaded once at startup (closure)
  - Feature column order loaded from the MLflow model signature (no hardcoding)
  - Feast features fetched via HTTP REST (/get-online-features)
  - score() applied to every message via sdf.apply()
  - Output published natively via sdf.to_topic()   ← no separate Producer

Data flow:
  [telemetry-data]  →  fetch Feast features  →  IsolationForest  →  [predictions]
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any
import json

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from quixstreams import Application
import numpy as np

from config import Config

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(Config.SERVICE_NAME)
logger.info("Workdir: %s", os.getcwd())   # BUG FIX: was logging.info (root logger)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """
    Load the sklearn Pipeline from the MLflow Model Registry and extract
    the feature column order from the model signature.

    The signature was saved during training with create_and_log_signature(),
    which used the raw (untransformed) DataFrame — so the column names and
    order here exactly match what the Pipeline expects at inference time.

    Returns
    -------
    (model_pipeline, model_uri_string, feature_columns_list)
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{Config.MLFLOW_MODEL_NAME}/{Config.MLFLOW_MODEL_STAGE}"

    logger.info(
        "Loading model from MLflow: %s  (tracking=%s)",
        model_uri, Config.MLFLOW_TRACKING_URI,
    )
    loaded = mlflow.sklearn.load_model(model_uri)

    # BUG FIX: feature column order was hardcoded — risky if the Feast join
    # or retrain.py column drops change the order between runs. Load it
    # directly from the MLflow signature so training and inference are always
    # in sync with zero manual maintenance.
    model_info     = mlflow.models.get_model_info(model_uri)
    feature_columns = [col.name for col in model_info.signature.inputs]
    logger.info("✓ Model loaded — %d features: %s", len(feature_columns), feature_columns)

    return loaded, model_uri, feature_columns


# ─────────────────────────────────────────────────────────────────────────────
# FEAST  —  HTTP REST helper
# ─────────────────────────────────────────────────────────────────────────────

def feast_get_online_features(
    session: requests.Session,
    machine_id: int,
) -> dict[str, Any]:
    """
    Fetch the full feature vector for *machine_id* from the Feast feature
    server via HTTP POST /get-online-features.

    The feature service 'machine_anomaly_service_v1' returns:
      • machine_streaming_features  (raw sensors + rolling windows)
      • machine_batch_features      (daily aggregations)

    Returns
    -------
    dict  {feature_name: value}   — None for features with status != PRESENT
    """
    payload = {
        "feature_service":    Config.FEAST_FEATURE_SERVICE,
        "full_feature_names": Config.FEAST_FULL_FEATURE_NAMES,
        "entities": {
            Config.FEAST_ENTITY_KEY: [machine_id],  # e.g. {"Machine_ID": [42]}
        },
    }

    url = f"{Config.FEAST_SERVER_URL.rstrip('/')}/get-online-features"
    response = session.post(url, json=payload, timeout=Config.FEAST_REQUEST_TIMEOUT_S)
    response.raise_for_status()

    data    = response.json()
    names   = data["metadata"]["feature_names"]   # list of feature name strings
    results = data["results"]                      # list of {values, statuses, ...}

    features: dict[str, Any] = {}
    for name, res in zip(names, results):
        status = (res.get("statuses") or ["MISSING"])[0]
        value  = (res.get("values")   or [None])[0]
        features[name] = value if status == "PRESENT" else None

    logger.info("Feast features: %s", features)   # BUG FIX: was logging.info (root logger)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_x(features: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """
    Order the feature dict into a one-row DataFrame whose columns match
    feature_columns exactly (the order the Pipeline was trained on, as
    read from the MLflow model signature).

    Type handling:
      - Cycle_Phase_ID is kept as str so the OneHotEncoder branch in the
        ColumnTransformer receives the same type it was trained on.
      - All other columns are coerced to numeric.
    """
    row = {col: features.get(col) for col in feature_columns}
    x   = pd.DataFrame([row], columns=feature_columns)

    x = x.replace({None: np.nan})

    for col in x.columns:
        if col == "Cycle_Phase_ID":
            # BUG FIX: pd.to_numeric would silently convert "1" → 1,
            # bypassing the OneHotEncoder and causing a schema mismatch.
            # retrain.py casts this to str; inference must do the same.
            x[col] = x[col].astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    return x


def load_thresholds(filepath: str = "outputs/thresholds.json") -> dict:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("File thresholds not found")
        raise

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, x: pd.DataFrame, threshold: float) -> tuple[int, float, int]:
    """
    Runs the IsolationForest model and classifies the result using a custom threshold.
    """
    raw_label = int(model.predict(x)[0])
    anomaly_score = float(model.decision_function(x)[0])

    # Replace the native model logic with a custom threshold:
    # If the score is LOWER than the threshold, it is considered an anomaly.
    is_anomaly = (
        Config.OUTPUT_ANOMALY
        if anomaly_score < threshold
        else Config.OUTPUT_NORMAL
    )

    return is_anomaly, anomaly_score, raw_label


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # 1. Load model + feature column order from MLflow
    model, model_uri, feature_columns = load_model()


    thresholds = load_thresholds()
    chosen_threshold = thresholds["p50"] # o p05, a seconda di quanto vuoi essere severo
    logger.info("Thresholds set to: %.4f", chosen_threshold)


    # 2. Open a persistent HTTP session for Feast REST calls
    session = requests.Session()

    # 3. QuixStreams application
    app = Application(
        broker_address=Config.KAFKA_BOOTSTRAP_SERVERS,
        consumer_group=Config.CONSUMER_GROUP,
        auto_offset_reset=Config.AUTO_OFFSET_RESET,
    )

    # 4. Topics
    t_in  = app.topic(Config.TOPIC_INPUT,  value_deserializer="json")   # telemetry-data
    t_out = app.topic(Config.TOPIC_OUTPUT, value_serializer="json")     # predictions

    # 5. Streaming dataframe
    sdf = app.dataframe(t_in)

    # ── Core inference function ───────────────────────────────────────────────
    def score(msg: dict[str, Any]) -> dict[str, Any]:
        """
        Full inference pipeline for a single telemetry message.

        Called by sdf.apply() for every record consumed from [telemetry-data].
        The result dict is serialised as JSON and published to [predictions]
        via sdf.to_topic() — no separate Kafka Producer needed.

        Output keys
        -----------
        machine_id         : canonical string id  (e.g. "M_0001")
        numeric_id         : int entity key used for Feast
        is_anomaly         : 1 = anomaly, 0 = normal
        anomaly_score      : IsolationForest decision_function value (< 0 = anomaly)
        isolation_label    : raw sklearn output (-1 or +1)
        features           : feature snapshot returned by Feast
        missing_features   : list of feature names that were None / not PRESENT
        source_timestamp   : original telemetry event timestamp
        scored_at          : UTC timestamp of this inference
        model_uri          : MLflow model URI used
        """
        # ── Build output skeleton (always populated, even on error) ──────────
        out = {
            "machine_id":       msg.get("Machine_ID"),
            "source_timestamp": msg.get("timestamp", ""),
            "scored_at":        datetime.now(timezone.utc).isoformat(),
            "model_uri":        model_uri,
        }

        try:
            # ── Parse Machine_ID → integer entity key ────────────────────────
            raw_id     = out["machine_id"]
            s          = str(raw_id).strip()
            numeric_id = int(s.replace("M_", "")) if s.startswith("M_") else int(s)

            out["numeric_id"] = numeric_id
            out["machine_id"] = f"M_{numeric_id:04d}"

            # ── Fetch features from Feast (Redis via HTTP) ───────────────────
            features = feast_get_online_features(session, numeric_id)

            missing = [k for k, v in features.items() if v is None]
            if missing:
                logger.warning("[%s] Missing features: %s", out["machine_id"], missing)

            # ── Build aligned feature DataFrame ─────────────────────────────
            x = build_x(features, feature_columns)

            # ── Run model ────────────────────────────────────────────────────
            is_anomaly, anomaly_score, raw_label = predict(model, x, chosen_threshold)

            # ── Enrich output ────────────────────────────────────────────────
            out["is_anomaly"]       = is_anomaly
            out["anomaly_score"]    = round(anomaly_score, 6)
            out["isolation_label"]  = raw_label
            out["features"]         = features
            out["missing_features"] = missing

            logger.info(
                "[%s] is_anomaly=%d  score=%.4f",
                out["machine_id"], is_anomaly, anomaly_score,
            )

        except Exception as exc:
            # Never crash the streaming loop — log and pass the error downstream
            out["error"] = str(exc)
            logger.error(
                "[%s] Inference failed: %s",
                out.get("machine_id", "?"), exc, exc_info=True,
            )

        return out

    # 6. Wire the pipeline
    sdf = sdf.apply(score)
    sdf = sdf.to_topic(t_out)

    logger.info(
        "Inference pipeline running: '%s' → '%s'  (feast=%s, model=%s)",
        Config.TOPIC_INPUT, Config.TOPIC_OUTPUT,
        Config.FEAST_SERVER_URL, model_uri,
    )

    # 7. Start the consumption loop (blocks until SIGTERM / SIGINT)
    app.run()   # BUG FIX: was app.run(sdf) — QuixStreams run() takes no arguments


if __name__ == "__main__":
    main()