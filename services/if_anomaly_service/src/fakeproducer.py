import time
import random
from datetime import datetime, timezone
from quixstreams import Application

# ==========================================
# CONFIGURATION
# ==========================================
# Use localhost:19092 if running outside Docker.
# Use redpanda:9092 if running inside a Docker container.
KAFKA_SERVER = "localhost:19092" 
TOPIC_NAME = "predictions" 

def generate_payload(machine_id: str, is_anomaly: int, numeric_id: int) -> dict:
    """Creates a fake prediction payload matching the exact ML model output."""
    
    now_utc = datetime.now(timezone.utc)
    
    # Simulate normal vs anomalous feature values and scores
    if is_anomaly == 1:
        anomaly_score = random.uniform(0.6, 0.95)
        isolation_label = -1 # Isolation Forest usually outputs -1 for anomalies
        vib_max = str(random.uniform(25.0, 40.0))
        curr_imb_mean = str(random.uniform(0.15, 0.3))
        curr_imb_ratio = str(random.uniform(0.2, 0.5))
        daily_vib_ratio = str(random.uniform(8.0, 15.0))
    else:
        anomaly_score = random.uniform(0.01, 0.15)
        isolation_label = 1 # 1 for normal
        vib_max = str(random.uniform(5.0, 10.0))
        curr_imb_mean = str(random.uniform(0.01, 0.05))
        curr_imb_ratio = str(random.uniform(0.02, 0.08))
        daily_vib_ratio = str(random.uniform(2.0, 4.0))

    return {
        "machine_id": machine_id,
        "source_timestamp": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "scored_at": now_utc.isoformat(),
        "model_uri": "models:/if_anomaly_detector/latest",
        "numeric_id": numeric_id,
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "isolation_label": isolation_label,
        "features": {
            "Machine_ID": numeric_id,
            "Vibration_RollingMax_10min": vib_max,
            "Current_Imbalance_RollingMean_5min": curr_imb_mean,
            "Current_Imbalance_Ratio": curr_imb_ratio,
            "Daily_Vibration_PeakMean_Ratio": daily_vib_ratio
        },
        "missing_features": []
    }

def main():
    app = Application(broker_address=KAFKA_SERVER)
    topic = app.topic(TOPIC_NAME, value_serializer='json')

    print(f"🚀 Starting Fake Producer...")
    print(f"📍 Target: {KAFKA_SERVER} | Topic: {TOPIC_NAME}")
    print("-" * 50)

    with app.get_producer() as producer:
        # Send 5 messages: the 4th one will be an anomaly
        for i in range(1, 6):
            is_anomaly = 1 if i == 4 else 0
            
            payload = generate_payload(
                machine_id="M_0001", 
                is_anomaly=is_anomaly,
                numeric_id=1
            )
            
            producer.produce(
                topic=topic.name,
                key=payload["machine_id"],
                value=payload
            )
            
            status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
            print(f"Sent [{status}]: Score: {payload['anomaly_score']:.4f} | VibMax: {float(payload['features']['Vibration_RollingMax_10min']):.2f}")
            
            time.sleep(2)

    print("-" * 50)
    print("🏁 Finished sending mock data.")

if __name__ == '__main__':
    main()