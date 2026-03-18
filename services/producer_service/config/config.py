import os
from pathlib import Path

class Config:
    """Internal configuration for the Producer Service"""
    
    # Kafka/Redpanda config
    KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'redpanda:9092')
    TOPIC_TELEMETRY = os.getenv('TOPIC_TELEMETRY', 'telemetry-data')
    
    # In Docker, we map the directory containing the dataset to /producer/data/streaming_data
    # So we build the path using the mounted volume.
    DATA_DIR = Path('/producer/data/streaming_data')
    STREAMING_DATASET_NAME = os.getenv('STREAMING_DATASET', 'industrial_washer_with_anomalies')
    
    # Absolute path to the file
    STREAMING_DATASET = DATA_DIR / STREAMING_DATASET_NAME

    BATCH_SIZE = 3
    BATCH_DELAY_SECONDS = 1
