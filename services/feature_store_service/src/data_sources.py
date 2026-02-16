"""
Data Sources for Feast Feature Store

This module defines the unified data sources.
Key Concept: The 'stream_source' includes a reference to 'machines_batch'.
This tells Feast: "When I ask for history, go to Parquet. When I ask for online, check what was pushed."
"""

from feast import PushSource
from feast.infra.offline_stores.file_source import FileSource

# 1. BATCH SOURCE (Historical / Offline)

machines_batch = FileSource(
    name="washing_batch_source",
    path="/feature_store_service/data/offline/machines_batch_features.parquet", 
    timestamp_field="event_timestamp", 
    description="Historical washing machine features stored in Parquet"
)

# 2. STREAMING SOURCE (Real-time / Online)

stream_source = PushSource(
    name="washing_stream_source",
    batch_source=machines_batch,
    description="Push source for real-time features from Redpanda"
)