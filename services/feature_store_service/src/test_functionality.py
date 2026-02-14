from feast import FeatureStore

# Initialize feature store
store = FeatureStore(repo_path="/feature_store_service")

# Get online features
features = store.get_online_features(
    features=[
        "machine_stream_features:Current_L1",
        "machine_stream_features:Vibration_mm_s"
    ],
    entity_rows=[
        {"Machine_ID": "1"},
        {"Machine_ID": "2"}
    ]
).to_dict()

# Or use feature service
features = store.get_online_features(
    feature_service="machine_anomaly_service_v1",
    entity_rows=[{"Machine_ID": "WASH_001"}]
).to_dict()

# Push features
from datetime import datetime
store.push(
    push_source_name="washing_stream_source",
    df=your_dataframe,
    to="online"  # or "offline" or "online_and_offline"
)