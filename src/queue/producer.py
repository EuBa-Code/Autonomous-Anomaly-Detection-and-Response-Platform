"""
RedPanda producer for streaming telemetry data
(RedPanda is Kafka API compatible, so the code remains the same)
"""
import time
import pandas as pd
import json
from confluent_kafka import Producer
from config import Config

def delivery_report(err, msg):
    """ Callback to confirm if the message reached RedPanda. """
    if err is not None:
        print(f"❌ Delivery failed: {err}")
    else:
        print(f"✅ Sent: {msg.topic()} [Partition: {msg.partition()}]")

def get_producer():
    """Create a RedPanda producer (uses Kafka protocol)."""
    conf = {
        'bootstrap.servers': Config.KAFKA_SERVER,  # RedPanda uses same Kafka protocol
        'linger.ms': 0,
        'acks': 1,
        'compression.type': 'snappy'  # Optional: RedPanda supports compression
    }
    return Producer(conf)

def start_streaming():
    producer = get_producer()

    try:
        df = pd.read_parquet(Config.STREAMING_DATASET)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"📡 Streaming to RedPanda started. Total records: {len(df)}")

    batch_size = 5  # Number of washing machines (messages) sent per batch

    for i in range(0, len(df), batch_size):
        # Take a group of 5 rows
        chunk = df.iloc[i : i + batch_size]

        for _, row in chunk.iterrows():
            data = row.to_dict()
            payload = json.dumps(data, default=str).encode("utf-8")

            # Send messages to RedPanda (buffered)
            producer.produce(
                Config.TOPIC_TELEMETRY,
                value=payload,
                callback=delivery_report
            )

        # Flush the buffer and send everything now
        producer.flush()

        print(f"🚀 Sent a batch of {len(chunk)} messages to RedPanda. Waiting 5 seconds...")
        time.sleep(5)

    print("✅ Streaming completed!")

if __name__ == "__main__":
    start_streaming()