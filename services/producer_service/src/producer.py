"""
Redpanda Producer — Streams telemetry data from Parquet to Redpanda.

Simulates real-time sensor data by reading from a Parquet file
and publishing messages to the telemetry-data topic in batches.
"""
import time
import json
import logging
import pandas as pd
from confluent_kafka import Producer
from config.config import Config

BATCH_SIZE = 3
BATCH_DELAY_SECONDS = 1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TelemetryProducer")


def delivery_report(err, msg):
    """Callback to confirm message delivery to Redpanda."""
    if err is not None:
        logger.error(f"Delivery failed: {err}")


def create_producer():
    """Create a Redpanda producer (Kafka protocol compatible)."""
    conf = {
        'bootstrap.servers': Config.KAFKA_SERVER,
        'linger.ms': 0,
        'acks': 1,
        'compression.type': 'snappy',
    }
    return Producer(conf)


def start_streaming():
    """Read Parquet file and stream records to Redpanda in batches."""
    producer = create_producer()

    try:
        df = pd.read_parquet(Config.STREAMING_DATASET)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    total_records = len(df)
    logger.info(f"Streaming {total_records} records to Redpanda topic '{Config.TOPIC_TELEMETRY}'")

    for i in range(0, total_records, BATCH_SIZE):
        chunk = df.iloc[i : i + BATCH_SIZE]

        for _, row in chunk.iterrows():
            payload = json.dumps(row.to_dict(), default=str).encode("utf-8")
            producer.produce(
                Config.TOPIC_TELEMETRY,
                value=payload,
                callback=delivery_report,
            )

        producer.flush()
        logger.info(f"Sent batch {i // BATCH_SIZE + 1} ({len(chunk)} messages)")
        time.sleep(BATCH_DELAY_SECONDS)

    logger.info(f"Streaming completed — {total_records} records sent")


if __name__ == "__main__":
    start_streaming()
