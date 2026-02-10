from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Float32
from src.entity import machine
from src.data_sources import stream_source, batch_source


#  Create the Feature View
#  A Feature View groups related features and defines how they are stored and served
machine_view = FeatureView(
    name="machine_stream_features",
    entities=[machine],
    ttl=timedelta(hours=24),  # Time-to-live for the features in the online store (Redis)
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
    source=stream_source,
    online=True  # Enables serving these features from the online store (e.g. Redis)
)

