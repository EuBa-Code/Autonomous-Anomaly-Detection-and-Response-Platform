import os


class Config:
    """Runtime configuration for the Streaming Service.

    All values can be overridden via environment variables so the same image
    runs identically in development (docker-compose) and production (k8s).
    """

    # ── Kafka / Redpanda ──────────────────────────────────────────────────────
    KAFKA_SERVER: str      = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
    TOPIC_TELEMETRY: str   = os.getenv("TOPIC_TELEMETRY", "telemetry-data")
    AUTO_OFFSET_RESET: str = os.getenv("AUTO_OFFSET_RESET", "latest")

    # ── Feast feature server ──────────────────────────────────────────────────
    FEAST_SERVER_URL: str  = os.getenv("FEAST_SERVER_URL", "http://feature_store_service:6566")

    # One push-source name per streaming window, matching data_sources.py
    PUSH_SOURCE_VIBRATION: str = os.getenv("PUSH_SOURCE_VIBRATION", "vibration_push_source")
    PUSH_SOURCE_CURRENT: str   = os.getenv("PUSH_SOURCE_CURRENT",   "current_push_source")

    # "online_and_offline" writes to Redis AND the Parquet backfill path.
    # Use "online" only during local development to skip the Parquet write.
    PUSH_TO: str = os.getenv("FEAST_PUSH_TO", "online_and_offline")

    # ── QuixStreams state (RocksDB) ───────────────────────────────────────────
    STATE_DIR: str = os.getenv("QUIX_STATE_DIR", "/tmp/quix_state")

    # ── Raw telemetry sink (entity DataFrame for point-in-time joins) ─────────
    # Every raw message is written here before any transformation so the batch
    # pipeline can build historical feature tables from the ground truth.
    ENTITY_DF_DIR: str    = os.getenv("ENTITY_DF_DIR", "/data/entity_df")
    ENTITY_DF_FORMAT: str = os.getenv("ENTITY_DF_FORMAT", "parquet")