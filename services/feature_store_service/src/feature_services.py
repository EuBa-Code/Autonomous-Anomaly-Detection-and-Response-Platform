"""
Feature Service Definitions for Feast Feature Store

Feature Services provide a convenient way to group features for specific use cases.
They act as a contract between feature producers and consumers (ML models).
"""

from feast import FeatureService 
from src import machine_features

# ============================================================================
# ANOMALY DETECTION SERVICE
# ============================================================================
# Groups all features needed for real-time anomaly detection
# Models can request features by referencing this service instead of 
# individual feature views
machine_feature_service_v1 = FeatureService(
    name="machine_anomaly_service_v1",
    features=[
        machine_features,  # All machine fatures (sensors + engineered)
    ],
    description="Feature service for washing machine anomaly detection model v1"
)