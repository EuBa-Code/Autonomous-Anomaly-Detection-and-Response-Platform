from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float32
from .entity import machine


# Define the data source (where Feast reads the historical features from)
#    This can be a Parquet or CSV file
machine_features_source = FileSource(
    path="/data/historical_processed/train.parquet",
    event_timestamp_column="event_timestamp"
)

#  Create the Feature View
#  A Feature View groups related features and defines how they are stored and served
machine_view = FeatureView(
    name="machine_features",
    entities=[machine],
    ttl=timedelta(hours=3),  # Time-to-live for the features in the online store (Redis)
    schema=[
        Field(name="Current_avg", dtype=Float32),
        Field(name="Apparent_Power", dtype=Float32),
        Field(name="Active_Power", dtype=Float32),
        Field(name="Reactive_Power", dtype=Float32),
        Field(name="Power_Factor", dtype=Float32),
        Field(name="THD_Current", dtype=Float32),
        Field(name="Current_P_to_P", dtype=Float32),
        Field(name="Max_Current_Instance", dtype=Float32),
        Field(name="Inrush_Peak", dtype=Float32),
        Field(name="Phase_imbalance", dtype=Float32),
        Field(name="Energy_per_Cycle_Wh", dtype=Float32),
    ],
    source=machine_features_source,
    online=True  # Enables serving these features from the online store (e.g. Redis)
    tags={"team": "ml", "project": "predictive_maintenance", "domain":"electrical_monitoring"}  # Optional metadata tags
)
