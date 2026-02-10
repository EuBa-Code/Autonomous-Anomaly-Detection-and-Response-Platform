from feast import FeatureService 
from features import machine_batch_view, machine_stream_view

# Create a feature service for the machine view 
machine_feature_service_v1 = FeatureService(
    name="machine_feature_service_v1",
    features=[
        machine_stream_view,
        machine_batch_view
    ],
)