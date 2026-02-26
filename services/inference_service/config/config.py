"""
Configuration for the Inference Pipeline Service.

All values are read from environment variables with sensible defaults,
so the service can be configured at container launch without rebuilding
the image.
"""

import os


class Config:
    # ── Redpanda / Kafka ──────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
    CONSUMER_GROUP:          str = os.getenv("CONSUMER_GROUP", "inference-pipeline-v1")
    AUTO_OFFSET_RESET:       str = os.getenv("AUTO_OFFSET_RESET", "earliest")

    # Input  : raw telemetry produced by producer.py
    TOPIC_INPUT:  str = os.getenv("TOPIC_INPUT",  "telemetry-data")

    # Output : enriched predictions (Quix serialises via sdf.to_topic)
    TOPIC_OUTPUT: str = os.getenv("TOPIC_OUTPUT", "predictions")

    # ── Feast feature server (HTTP REST) ──────────────────────────────────────
    # Point to the `feast serve` container, e.g. http://feast-server:6566
    FEAST_SERVER_URL:          str  = os.getenv("FEAST_SERVER_URL",      "http://feature_store_service:6566")
    FEAST_FEATURE_SERVICE:     str  = os.getenv("FEAST_FEATURE_SERVICE", "machine_anomaly_service_v1")
    FEAST_ENTITY_KEY:          str  = os.getenv("FEAST_ENTITY_KEY",      "Machine_ID")
    FEAST_FULL_FEATURE_NAMES:  bool = os.getenv("FEAST_FULL_FEATURE_NAMES", "false").lower() == "true"
    FEAST_REQUEST_TIMEOUT_S:   int  = int(os.getenv("FEAST_REQUEST_TIMEOUT_S", "5"))

    # ── MLflow Model Registry ─────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MLFLOW_MODEL_NAME:   str = os.getenv("MLFLOW_MODEL_NAME",   "if_anomaly_detector")

    # Stage can be "latest", "Production", "Staging", or a version number
    MLFLOW_MODEL_STAGE:  str = os.getenv("MLFLOW_MODEL_STAGE",  "latest")

    # ── IsolationForest label convention ─────────────────────────────────────
    # sklearn IsolationForest: -1 = anomaly,  +1 = normal
    ISOLATION_FOREST_ANOMALY_CLASS: int = -1

    # Downstream (output payload): 1 = anomaly,  0 = normal
    OUTPUT_ANOMALY: int = 1
    OUTPUT_NORMAL:  int = 0

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL:  str = os.getenv("LOG_LEVEL",  "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ── Service metadata ──────────────────────────────────────────────────────
    SERVICE_NAME:    str = "inference-pipeline"
    SERVICE_VERSION: str = "1.0.0"