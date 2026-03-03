"""
Redpanda Producer — Streams telemetry data from Parquet to Redpanda using quixstreams.

Simulates real-time sensor data by reading from a Parquet file
and publishing messages to the telemetry-data topic in batches.
"""
import time
import logging
import pandas as pd
from quixstreams import Application
from config.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TelemetryProducer")

app = Application(broker_address=Config.KAFKA_SERVER, consumer_group=Config.TOPIC_TELEMETRY)
topic_telemetry = app.topic(Config.TOPIC_TELEMETRY, value_serializer='json')

def main():
    logger.info(f'PRODUCER: Reading parquet file: {Config.STREAMING_DATASET}')

    try:
        df = pd.read_parquet(Config.STREAMING_DATASET)
    except Exception as e:
        logger.error(f"Error while reading the Parquet file: {e}")
        return

    total_rows = len(df)
    logger.info(f"File loaded. Total rows to send: {total_rows}")

    with app.get_producer() as producer:
        counter = 0
        
        for _, row in df.iterrows():
            msg = topic_telemetry.serialize(
                key=str(row['Machine_ID']),
                value=row.to_dict()
            )

            # Send single message
            producer.produce(
                topic=topic_telemetry.name,
                value=msg.value,
                key=msg.key  # Preserves message ordering
            )
            
            counter += 1

            # Batch delay handling (Real-time simulation)
            if counter % Config.BATCH_SIZE == 0:
                logger.info(f"Sent batch of {Config.BATCH_SIZE} messages. Waiting...")
                time.sleep(Config.BATCH_DELAY_SECONDS)

            # Progress log every 100 messages
            if counter % 100 == 0:
                logger.info(f'Progress: {counter}/{total_rows} messages sent.')

    logger.info('Dataset fully uploaded to Redpanda!')

if __name__ == '__main__':
    main()