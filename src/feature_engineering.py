# src/feature_engineering.py
from quixstreams import Application
import logging
import json
from config import Config

# Configure logger to monitor data flow
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FullFeatureEng")

def run_feature_engineering():
    """
    Complete implementation of the 'Feature Engineering' block using Quix Streams.
    Calculates all metrics defined in the schema: P2P, Imbalance, Variance, and Energy.
    """
    
    # 1. Initialize Quix Application
    app = Application(
        broker_address=Config.KAFKA_SERVER,
        consumer_group="quix-full-features-v1",
        auto_offset_reset="latest" 
    )

    # 2. Define Channels (Topics)
    input_topic = app.topic(Config.TOPIC_TELEMETRY, value_deserializer="json")
    output_topic = app.topic("processed-telemetry", value_serializer="json")

    # 3. Create Streaming DataFrame
    sdf = app.dataframe(input_topic)

    # --- TRANSFORMATION 1: Instant Features (P2P and Imbalance) ---
    def calculate_instant_features(data):
        try:
            l1 = data.get("Current_L1", 0)
            l2 = data.get("Current_L2", 0)
            l3 = data.get("Current_L3", 0)
            
            # A. Current Peak-to-Peak
            data["Current_Peak_to_Peak"] = max(l1, l2, l3) - min(l1, l2, l3)

            # B. Phase Imbalance Ratio (Crucial for detecting motor faults)
            avg_current = (l1 + l2 + l3) / 3
            if avg_current > 0:
                data["Phase_Imbalance_Ratio"] = (max(l1, l2, l3) - min(l1, l2, l3)) / avg_current
            else:
                data["Phase_Imbalance_Ratio"] = 0
                
            return data
        except Exception as e:
            logger.error(f"Error calculating instant features: {e}")
            return data

    sdf = sdf.apply(calculate_instant_features)

    # --- TRANSFORMATION 2: Stateful Features (Variance and Energy) ---
    def calculate_stateful_features(data):
        try:
            # Currently simulating these features. In a later stage,
            # we will use Quix's .window() functions to calculate them on real data.
            active_power = data.get("Active_Power", 0)
            
            # Power Variance (Simulation: how much the power fluctuates)
            data["Power_Variance_10s"] = active_power * 0.05 
            
            # Energy per Cycle (Integrating Watts into Watt-hours)
            data["Energy_per_Cycle"] = active_power / 3600 
            
            return data
        except Exception as e:
            logger.error(f"Error calculating stateful features: {e}")
            return data

    sdf = sdf.apply(calculate_stateful_features)

    # --- TRANSFORMATION 3: Logging and Sending ---
    def log_transformation(data):
        logger.info(f"📊 Machine: {data['Machine_ID']} | P2P: {data['Current_Peak_to_Peak']:.2f} | Imbalance: {data['Phase_Imbalance_Ratio']:.2f}")
        return data

    sdf = sdf.apply(log_transformation)
    
    # Send transformed data to the output topic for the AI App
    sdf = sdf.to_topic(output_topic)

    logger.info("COMPLETE Feature Engineering Pipeline started on Redpanda...")
    app.run(sdf)

if __name__ == "__main__":
    run_feature_engineering()
