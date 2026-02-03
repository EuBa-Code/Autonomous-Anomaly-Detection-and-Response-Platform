# src/feature_engineering.py
from quixstreams import Application
import logging
import joblib
import pandas as pd
from config import Config

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamingPreprocessor")

def run_streaming_preprocessing():
    """
    Real-time preprocessing using Quix Streams.
    Loads a pre-trained Scaler/Encoder and applies it to incoming data.
    """
    
    # 1. Load Preprocessor Artifact
    preprocessor_path = Config.ROOT_DIR / "models" / "preprocessor.joblib"
    
    if not preprocessor_path.exists():
        logger.error(f"Preprocessor artifact not found at {preprocessor_path}. Please run src/train.py first.")
        return

    logger.info("Loading pre-trained Scaler and Encoder...")
    preprocessor = joblib.load(preprocessor_path)

    # 2. Initialize Quix Application
    app = Application(
        broker_address=Config.KAFKA_SERVER,
        consumer_group="quix-preprocessing-v1",
        auto_offset_reset="latest" 
    )

    input_topic = app.topic(Config.TOPIC_TELEMETRY, value_deserializer="json")
    output_topic = app.topic("processed-telemetry", value_serializer="json")

    sdf = app.dataframe(input_topic)

    # 3. Apply Transformations
    def apply_scaling_and_ohe(data):
        try:
            # Keep identifiers for logging/tracking
            machine_id = data.get("Machine_ID", "Unknown")
            timestamp = data.get("timestamp", "Unknown")
            
            # Convert single message to DataFrame for the preprocessor
            df_row = pd.DataFrame([data])
            
            # Apply Normalization and One-Hot Encoding
            # This uses the exact same mean/std/categories from the training set
            transformed_vector = preprocessor.transform(df_row)
            
            # Create a clean dictionary for the output
            # We send the processed features + identifiers
            result = {
                "Machine_ID": machine_id,
                "timestamp": timestamp,
                "features": transformed_vector[0].tolist() # Convert numpy array to list for JSON
            }
            
            logger.info(f"Preprocessed data for {machine_id} (Vector size: {transformed_vector.shape[1]})")
            return result
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return None

    # Filter out None values and apply transformation
    sdf = sdf.apply(apply_scaling_and_ohe)
    sdf = sdf.filter(lambda x: x is not None)
    
    # 4. Send to output topic
    sdf = sdf.to_topic(output_topic)

    logger.info("Streaming Preprocessing Pipeline (Scaling + OHE) started...")
    app.run(sdf)
