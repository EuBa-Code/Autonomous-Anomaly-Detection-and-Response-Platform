from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float32

# Define the entity (unique identifier for each machine)
machine = Entity(
    name="machine_id",
    join_keys=["machine_id"]
)

# Define the data source (where Feast reads the historical features from)
#    This can be a Parquet or CSV file
source = FileSource(
    path="/app/data/dataset.parquet",
    event_timestamp_column="event_timestamp"
)

#  Create the Feature View
#  A Feature View groups related features and defines how they are stored and served
machine_view = FeatureView(
    name="machine_features",
    entities=[machine],
    schema=[
        Field(name="cpu_usage", dtype=Float32),
        Field(name="temperature", dtype=Int64),
    ],
    source=source,
    online=True  # Enables serving these features from the online store (e.g. Redis)
)
