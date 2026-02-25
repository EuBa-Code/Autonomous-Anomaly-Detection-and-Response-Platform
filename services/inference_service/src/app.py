import logging
import os
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
from confluent_kafka import Producer
import uvicorn
import time
from config import Config

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InferenceService")

app = FastAPI(
    title="Washing Machine Anomaly Detection Inference API",
    description="Real-time anomaly detection using Isolation Forest and Feast Online Store (Redis)",
    version="1.0.0"
)

# Global State
model = None
store = None
kafka_producer = None
FEATURE_VIEW_NAME = "machine_streaming_features"

class PredictionRequest(BaseModel):
    machine_id: str

class PredictionResponse(BaseModel):
    machine_id: str
    is_anomaly: int
    anomaly_score: float
    model_version: str = "v1"

def retry_with_backoff(func, max_retries=5, initial_delay=2, backoff_factor=2):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Callable to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries
    
    Returns:
        Result of the function call
    
    Raises:
        Exception: If all retries are exhausted
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries} attempts failed.")
    
    raise last_exception

@app.on_event("startup")
def startup_event():
    """Load model artifacts and initialize Feast connection on startup."""
    global model, store, kafka_producer
    
    try:
        # 1. Initialize Kafka Producer (early, no external dependencies)
        logger.info("Initializing Kafka Producer...")
        try:
            kafka_producer = Producer({
                'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'redpanda:9092'),
                'acks': 1,
            })
            logger.info("✓ Kafka Producer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kafka Producer: {e}. Continuing without it.")
            kafka_producer = None

        # 2. Load Model from MLflow Registry with retry logic
        logger.info("Connecting to MLflow Tracking Server...")
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        def load_model_from_mlflow():
            client = mlflow.tracking.MlflowClient()
            
            model_name = "if_anomaly_detector"
            logger.info(f"Fetching latest version of model '{model_name}' from MLflow...")
            
            versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            
            if not versions:
                raise RuntimeError(f"No registered model found for '{model_name}'")
                
            versions.sort(key=lambda x: int(x.version), reverse=True)
            latest_version = versions[0].version
            
            model_uri = f"models:/{model_name}/{latest_version}"
            logger.info(f"Loading model from {model_uri}...")
            
            loaded_model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✓ Model loaded successfully (Version {latest_version})")
            return loaded_model
        
        try:
            model = retry_with_backoff(
                load_model_from_mlflow,
                max_retries=5,
                initial_delay=3,
                backoff_factor=2
            )
        except Exception as e:
            logger.error(f"Failed to load model from MLflow after retries: {e}")
            logger.warning("Falling back to local file system model if available...")
            model_path = Config.MODEL_PATH
            if model_path.exists():
                model = joblib.load(model_path)
                logger.info("✓ Fallback: Local model loaded successfully")
            else:
                raise RuntimeError(
                    f"Could not load model from MLflow or local filesystem. "
                    f"MLflow error: {e}, Local path: {model_path}"
                )

        # 3. Initialize Feast
        repo_path = os.getenv("FEAST_REPO_PATH", "/inference_service")
        logger.info(f"Initializing Feast Feature Store from {repo_path}...")
        store = FeatureStore(repo_path=repo_path)
        logger.info("✓ Feast Feature Store connection established")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise e

@app.post("/predict", response_model=PredictionResponse)
def predict_anomaly(request: PredictionRequest):
    """
    Predict anomaly status for a given machine_id based on real-time features from Redis.
    """
    if not store or not model:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    machine_id = request.machine_id
    
    # 0. Convert 'M_0010' to integer '10' for Feast Entity matching
    try:
        if machine_id.startswith("M_"):
            numeric_id = int(machine_id.replace("M_", ""))
        else:
            numeric_id = int(machine_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid machine_id format: {machine_id}. Expected format like 'M_0010' or '10'")

    # 1. Fetch Features from Feast using FeatureService
    try:
        logger.debug(f"Fetching online features via FeatureService for numeric_id={numeric_id}")
        
        response = store.get_online_features(
            features=store.get_feature_service("machine_anomaly_service_v1"),
            entity_rows=[{"Machine_ID": numeric_id}]
        )
        
        data_dict = response.to_dict()
        df = pd.DataFrame(data_dict)
        
        # 1.5 Map Feast columns to MLflow required features
        try:
            feature_columns = list(model.feature_names_in_)
            df_features = df[feature_columns]
        except AttributeError:
             logger.error("Model does not expose feature_names_in_")
             raise HTTPException(status_code=500, detail="Cannot determine required features from Model Pipeline")
        except KeyError as e:
             logger.error(f"Feature Store returned incomplete data for Model. Missing: {e}")
             raise HTTPException(status_code=500, detail=f"Feature/Model Mismatch: {e}")

        if df_features.isnull().values.any():
            if df_features.isnull().all().all():
                logger.warning(f"No features found for machine_id={machine_id} (numeric {numeric_id})")
                raise HTTPException(status_code=404, detail="Machine not found or offline")
            else:
                missing_cols = df_features.columns[df_features.isnull().any()].tolist()
                logger.warning(f"Partial features found for machine_id={machine_id} (numeric {numeric_id}). Missing exactly: {missing_cols}")
                logger.debug(f"Full Feast payload received: {data_dict}")
                raise HTTPException(status_code=422, detail=f"Incomplete feature data available. Missing exactly: {missing_cols}")

    except Exception as e:
        logger.error(f"Error fetching features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Predict
    try:
        prediction_code = model.predict(df_features)[0]
        score = model.decision_function(df_features)[0]
        
        # Isolation Forest: -1 (Anomaly), 1 (Normal)
        # Manteniamo -1/1 su Redpanda per l'AnomalyMonitor
        # Convertiamo a 0/1 solo per la response HTTP
        is_anomaly_http = 1 if prediction_code == -1 else 0
        
        logger.info(f"Prediction for Machine {machine_id}: Is_Anomaly={is_anomaly_http}, Score={score:.4f}")

        # 3. Pubblica su Redpanda con -1/1 (raw Isolation Forest output)
        if kafka_producer:
            try:
                payload = {
                    "machine_id": machine_id,
                    "is_anomaly": int(prediction_code),  # -1 o 1, raw output
                    "anomaly_score": float(score)
                }
                kafka_producer.produce(
                    "predictions",
                    value=json.dumps(payload).encode("utf-8")
                )
                kafka_producer.flush()
                logger.info(f"Published prediction to Redpanda: {payload}")
            except Exception as e:
                logger.warning(f"Failed to publish prediction to Kafka: {e}")
                # Don't fail the prediction if Kafka is down
        
        # 4. Ritorna la response HTTP con 0/1
        return {
            "machine_id": machine_id,
            "is_anomaly": is_anomaly_http,
            "anomaly_score": float(score)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Model Prediction Error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint for liveness probe."""
    if not store or not model:
        return {"status": "unhealthy", "detail": "Service not initialized"}
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)