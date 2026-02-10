from feast import FeatureService 
from features import machine_view

# Create a feature service for the machine view 
machine_feature_service_v1 = FeatureService(
    name="machine_anomaly_service_v1",
    features=[
        machine_view,
    ],
)