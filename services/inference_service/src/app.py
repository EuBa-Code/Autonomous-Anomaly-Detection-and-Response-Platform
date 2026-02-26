"""
Inference Pipeline — Washing Machine Anomaly Detection
======================================================

  - Model loaded once at startup (closure)
  - Feast features fetched via HTTP REST (/get-online-features)
  - score() applied to every message via sdf.apply()
  - Output published natively via sdf.to_topic()   ← no separate Producer

Data flow:
  [telemetry-data]  →  fetch Feast features  →  IsolationForest  →  [predictions]
"""

import logging
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from quixstreams import Application

from config import Config

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(Config.SERVICE_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN ORDER
# Must match the exact order the sklearn Pipeline was trained on.
# These are the columns returned by machine_anomaly_service_v1.
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # ── Raw sensor readings (machine_streaming_features) ─────────────────────
    "Cycle_Phase_ID",
    "Current_L1",
    "Current_L2",
    "Current_L3",
    "Voltage_L_L",
    "Water_Temp_C",
    "Motor_RPM",
    "Water_Flow_L_min",
    "Vibration_mm_s",
    "Water_Pressure_Bar",
    # ── Streaming-derived features (computed by QuixStreams pipeline) ─────────
    "Current_Imbalance_Ratio",
    "Vibration_RollingMax_10min",
    "Current_Imbalance_RollingMean_5min",
    # ── Batch-derived features (computed by PySpark batch pipeline) ───────────
    "Daily_Vibration_PeakMean_Ratio",
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """
    Load the sklearn Pipeline from the MLflow Model Registry.
    Called once at startup — the model object is then captured in the
    score() closure and reused for every message.

    Returns
    -------
    (model_pipeline, model_uri_string)
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{Config.MLFLOW_MODEL_NAME}/{Config.MLFLOW_MODEL_STAGE}"

    logger.info(
        "Loading model from MLflow: %s  (tracking=%s)",
        model_uri, Config.MLFLOW_TRACKING_URI,
    )
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("✓ Model loaded")
    return model, model_uri


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
        "feature_service":     Config.FEAST_FEATURE_SERVICE,
        "full_feature_names":  Config.FEAST_FULL_FEATURE_NAMES,
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
        # zip produces: ("Vibration_mm_s", {"values": [1.23], "statuses": ["PRESENT"]}, ...)
        status  = (res.get("statuses") or ["MISSING"])[0]
        value   = (res.get("values")   or [None])[0]
        features[name] = value if status == "PRESENT" else None

    return features


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_x(features: dict[str, Any]) -> pd.DataFrame:
    """
    Order the feature dict into a one-row DataFrame whose columns match
    FEATURE_COLUMNS exactly (the order the Pipeline was trained on).
    Numeric coercion ensures no accidental string types slip through.
    """
    row = {col: features.get(col) for col in FEATURE_COLUMNS}
    x   = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="ignore")

    return x


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, x: pd.DataFrame) -> tuple[int, float]:
    """
    Run the IsolationForest pipeline and return (is_anomaly, anomaly_score).

      is_anomaly    : 1 = anomaly,  0 = normal    (downstream-friendly convention)
      anomaly_score : decision_function value;  < 0 → anomaly territory
    """
    raw_label    = int(model.predict(x)[0])          # -1 (anomaly) or +1 (normal)
    anomaly_score = float(model.decision_function(x)[0])

    is_anomaly = (
        Config.OUTPUT_ANOMALY
        if raw_label == Config.ISOLATION_FOREST_ANOMALY_CLASS
        else Config.OUTPUT_NORMAL
    )
    return is_anomaly, anomaly_score, raw_label


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # 1. Load model once at startup
    model, model_uri = load_model()

    # 2. Open a persistent HTTP session for Feast REST calls
    session = requests.Session()

    # 3. Quix Streams application
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
        machine_id         : canonical string id  (e.g. "M_0042")
        numeric_id         : int entity key used for Feast
        is_anomaly         : 1 = anomaly, 0 = normal
        anomaly_score      : IsolationForest decision_function value
        isolation_label    : raw sklearn output (-1 or +1)
        features           : feature snapshot returned by Feast
        missing_features   : list of feature names that were None / not PRESENT
        source_timestamp   : original telemetry event timestamp
        scored_at          : UTC timestamp of this inference
        model_uri          : MLflow model URI used
        """
        # ── Build output skeleton (always populated, even on error) ──────────
        out = {
            "machine_id":        msg.get("Machine_ID") or msg.get("machine_id"),
            "source_timestamp":  msg.get("timestamp", ""),
            "scored_at":         datetime.now(timezone.utc).isoformat(),
            "model_uri":         model_uri,
        }

        try:
            # ── Parse Machine_ID → integer entity key ────────────────────────
            raw_id = out["machine_id"]
            s = str(raw_id).strip()
            numeric_id = int(s.replace("M_", "")) if s.startswith("M_") else int(s)

            # Store canonical forms in the output
            out["numeric_id"]  = numeric_id
            out["machine_id"]  = f"M_{numeric_id:04d}"

            # ── Fetch features from Feast (Redis via HTTP) ───────────────────
            features = feast_get_online_features(session, numeric_id)

            # Track which features are missing (None / not PRESENT in Redis)
            missing = [k for k, v in features.items() if v is None]
            if missing:
                logger.warning("[%s] Missing features: %s", out["machine_id"], missing)

            # ── Build aligned feature DataFrame ─────────────────────────────
            x = build_x(features)

            # ── Run model ────────────────────────────────────────────────────
            is_anomaly, anomaly_score, raw_label = predict(model, x)

            # ── Enrich output ────────────────────────────────────────────────
            out["is_anomaly"]      = is_anomaly        # 1 / 0
            out["anomaly_score"]   = round(anomaly_score, 6)
            out["isolation_label"] = raw_label          # -1 / +1 (raw sklearn)
            out["features"]        = features           # full Feast snapshot
            out["missing_features"]= missing

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
    sdf = sdf.apply(score)      # run inference on every telemetry message
    sdf = sdf.to_topic(t_out)   # publish result to [predictions] (Quix handles serialisation)

    logger.info(
        "Inference pipeline running: '%s' → '%s'  (feast=%s, model=%s)",
        Config.TOPIC_INPUT, Config.TOPIC_OUTPUT,
        Config.FEAST_SERVER_URL, model_uri,
    )

    # 7. Start the consumption loop (blocks until SIGTERM / SIGINT)
    app.run(sdf)


if __name__ == "__main__":
    main()