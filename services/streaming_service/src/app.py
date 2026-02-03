"""
Anomaly Detection App using RedPanda
(RedPanda is Kafka API compatible, so the code remains the same)
"""
import json
import pandas as pd
import joblib  # Standard library for loading AI models
from confluent_kafka import KafkaError
from config import Config
from src.queue import get_consumer, TelemetryData  # Import helper functions
from pydantic import ValidationError

# Load your trained AI Model
# MODEL_PATH = Config.ROOT_DIR / "models" / "isolation_forest.pkl"

def run_anomaly_detection():
    consumer = get_consumer()
    consumer.subscribe([Config.TOPIC_TELEMETRY])
    print(f"🧠 Anomaly Detector running on RedPanda topic: {Config.TOPIC_TELEMETRY}...\n")

    try:
        while True:
            # Get Message from RedPanda
            msg = consumer.poll(1.0)
            if msg is None: 
                continue
            if msg.error():
                print(f"❌ Error: {msg.error()}")
                continue

            # Decode & Validate (Pydantic)
            try:
                data_dict = json.loads(msg.value().decode('utf-8'))
                telemetry = TelemetryData(**data_dict)  # This validates types automatically
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"⚠️ Bad Data: {e}")
                continue

            # Pre-process for AI
            # Convert Pydantic model to DataFrame
            df_row = pd.DataFrame([telemetry.model_dump()])
                       
            # 4. PREDICT
            # if model:
            #     prediction = model.predict(features)
            #     score = model.decision_function(features) # For Isolation Forest
            #     
            #     # -1 is Anomaly, 1 is Normal (usually)
            #     if prediction[0] == -1:
            #         print(f"🚨 ANOMALY DETECTED for {telemetry.Machine_ID}!")
            #         print(f"   Score: {score[0]}")
            #     else:
            #         print(f"✅ Status OK (Score: {score[0]})")
            
            # For now, just print the clean features to prove it works
            all_features = telemetry.model_dump()
            
            # Convert to DataFrame for the AI model
            df_row = pd.DataFrame([all_features])
            
            # Print all features
            print(f"🔍 Analyzing Machine: {telemetry.Machine_ID}")
            print("-" * 30)
            for feature, value in all_features.items():
                print(f"{feature:25}: {value}")
            print("-" * 30 + "\n")


    except KeyboardInterrupt:
        print("\n🛑 Stopping anomaly detector...")
    finally:
        consumer.close()
        print("✅ Consumer closed gracefully")

if __name__ == "__main__":
    run_anomaly_detection()