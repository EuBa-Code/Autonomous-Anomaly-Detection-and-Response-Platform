import os
from pathlib import Path

class Config:
    """Internal configuration for the Streaming Service"""
    
    # Kafka/Redpanda config
    KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'redpanda:9092')
    TOPIC_TELEMETRY = os.getenv('TOPIC_TELEMETRY', 'telemetry-data')

    FEAST_SERVER_URL = 'http://feature_store_service:6566'
    PUSH_SOURCE_NAME = 'washing_stream_push'
    PUSH_TO = 'online_and_offline'

    AUTO_OFFSET_RESET = "earliest"

    STATE_DIR: str = "/tmp/quix_state"

    DATALAKE_DIR: str = "/app/datalake"

    DATALAKE_FORMAT: str = "parquet" 

