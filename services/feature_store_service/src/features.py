"""
Feature View Definitions for Feast Feature Store

This module defines feature views that group related features together.
Feature views specify:
- Which features to track
- How long to retain them (TTL)
- Which entity they belong to
- Which data source to use
"""

from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int64
from src import machine
from src import machines_batch, stream_source


machine_stream_features = FeatureView(
    name="machine_stream_features",
    entities=[machine],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="Cycle_Phase_ID", dtype=Int64),
        Field(name="Current_L1", dtype=Float32),
        Field(name="Current_L2", dtype=Float32),
        Field(name="Current_L3", dtype=Float32),
        Field(name="Voltage_L_L", dtype=Float32),
        Field(name="Water_Temp_C", dtype=Float32),
        Field(name="Motor_RPM", dtype=Float32),
        Field(name="Water_Flow_L_min", dtype=Float32),
        Field(name="Vibration_mm_s", dtype=Float32),
        Field(name="Water_Pressure_Bar", dtype=Float32),
        Field(name="Vibration_RollingMax_10min", dtype=Float32),
    ],
    source=stream_source, 
)

machines_batch_features = FeatureView(
    name="machine_batch_features",
    entities=[machine],
    ttl=timedelta(days=365),
    schema=[
        Field(name="Cycle_Phase_ID", dtype=Int64),
        Field(name="Current_L1", dtype=Float32),
        Field(name="Current_L2", dtype=Float32),
        Field(name="Current_L3", dtype=Float32),
        Field(name="Voltage_L_L", dtype=Float32),
        Field(name="Water_Temp_C", dtype=Float32),
        Field(name="Motor_RPM", dtype=Float32),
        Field(name="Water_Flow_L_min", dtype=Float32),
        Field(name="Vibration_mm_s", dtype=Float32),
        Field(name="Water_Pressure_Bar", dtype=Float32),
        Field(name="Vibration_RollingMax_10min", dtype=Float32),
    ],
    source=machines_batch, 
)

