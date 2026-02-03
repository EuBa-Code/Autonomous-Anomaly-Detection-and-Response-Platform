"""
RedPanda consumer with flexible telemetry data model
(RedPanda is Kafka API compatible, so the code remains the same)
"""
import json
import pandas as pd
from confluent_kafka import Consumer
from pydantic import BaseModel
from typing import Optional

from config import Config

# ------------------------------
# Updated Pydantic model based on actual data structure
# ------------------------------
class TelemetryData(BaseModel):
    # --- Identifiers ---
    timestamp: str
    Machine_ID: str
    Cycle_Phase_ID: Optional[int] = None

    # --- Electrical Metrics (L1, L2, L3) ---
    Current_L1: Optional[float] = None
    Current_L2: Optional[float] = None
    Current_L3: Optional[float] = None
    Voltage_L_L: Optional[float] = None
    
    # --- Power Metrics ---
    Active_Power: Optional[float] = None
    Reactive_Power: Optional[float] = None
    Power_Factor: Optional[float] = None
    Power_Variance_10s: Optional[float] = None
    Energy_per_Cycle: Optional[float] = None

    # --- Quality & Anomalous Indicators ---
    THD_Current: Optional[float] = None
    THD_Voltage: Optional[float] = None
    Current_Peak_to_Peak: Optional[float] = None
    Inrush_Current_Peak: Optional[float] = None
    Inrush_Duration: Optional[float] = None
    Phase_Imbalance_Ratio: Optional[float] = None

    # --- Labels (Optional during real-time inference) ---
    Is_Anomaly: Optional[int] = None
    Anomaly_Type: Optional[str] = None

    class Config:
        extra = "allow"

# ------------------------------
# RedPanda consumer setup (uses Kafka API)
# ------------------------------
def get_consumer() -> Consumer:
    """Create a RedPanda consumer to read telemetry messages."""
    conf = {
        'bootstrap.servers': Config.KAFKA_SERVER,  
        'group.id': 'anomaly-detector-v1',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'auto.commit.interval.ms': 5000
    }
    return Consumer(conf)