import logging
import pandas as pd
from quixstreams import Application
from config.config import Config

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamingService")

from feast import FeatureStore
import os

def run_streaming_service():
    """
    Real-time feature engineering service using Quix Streams.
    It performs:
    1. Consumption from Redpanda
    2. Real-time metrics calculation (Rolling Windows)
    3. Production of enriched features to 'processed-telemetry' (Unscaled)
    4. Push to Feast Online Store (Redis)
    """
    
    # 1. Initialize Feast and Quix
    repo_path = os.getenv("FEAST_REPO_PATH", "/streaming_service")
    try:
        store = FeatureStore(repo_path=repo_path)
        logger.info("Feast Feature Store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Feast: {e}")
        return

    app = Application(
        broker_address=Config.KAFKA_SERVER,
        consumer_group="quix-streaming-processor-v2",
        auto_offset_reset="earliest"
    )

    input_topic = app.topic(Config.TOPIC_TELEMETRY, value_deserializer="json")
    output_topic = app.topic("processed-telemetry", value_serializer="json")

    sdf = app.dataframe(input_topic)

    # 2. Feature Engineering & Redis Push
    def process_and_push(data):
        try:
            # Metadata
            machine_id = data.get("Machine_ID", "Unknown")
            timestamp_str = data.get("timestamp", "Unknown")

            # Simple simulation of "complex" real-time metric
            vibration = data.get("Vibration_mm_s", 0)
            data["Vibration_RollingMax_10min"] = vibration * 1.05 # Aggregation placeholder
            
            # Prepare for Feast: Convert to DataFrame
            df_row = pd.DataFrame([data])
            
            # 3. PUSH TO FEAST (Online Store)
            # Create a DataFrame that matches the Feast FeatureView schema
            # We must ensure the timestamp is a datetime object
            feast_df = df_row.copy()
            feast_df['event_timestamp'] = pd.to_datetime(timestamp_str)
            
            # Push to the PushSource defined in Feast
            store.push("washing_stream_source", feast_df)
            
            logger.info(f"Processed and pushed telemetry for {machine_id} to Redis")

            # Build result for next topic (optional debug)
            result = {
                "Machine_ID": machine_id,
                "timestamp": timestamp_str,
                "data": data
            }
            return result
            
        except Exception as e:
            logger.error(f"Error processing/pushing message: {e}")
            return None

    # Apply transformation
    sdf = sdf.apply(process_and_push)
    sdf = sdf.filter(lambda x: x is not None)
    
    # 4. Sink to Redpanda (Debug Topic)
    sdf = sdf.to_topic(output_topic)

    logger.info("Quix Streaming Service started successfully.")
    app.run(sdf)

if __name__ == "__main__":
    run_streaming_service()
