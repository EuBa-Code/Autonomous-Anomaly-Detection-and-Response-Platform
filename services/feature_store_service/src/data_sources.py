"""
Data Sources for Feast Feature Store

This module defines the unified data sources.
Key Concept: The 'stream_source' includes a reference to 'machines_batch'.
This tells Feast: "When I ask for history, go to Parquet. When I ask for online, check what was pushed."

IMPORTANT: This version is configured for PySpark partitioned parquet files
(multiple part-*.parquet files in a directory)
"""

from feast import PushSource
from feast.infra.offline_stores.file_source import FileSource


machines_batch = FileSource(
    name="washing_batch_source",
    path="/feature_store_service/data/offline/", 
    description="Historical washing machine features stored in partitioned Parquet files"
)


stream_source = PushSource(
    name="washing_stream_source",
    batch_source=machines_batch,
    description="Push source for real-time features from Redpanda"
)