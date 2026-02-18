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

### BATCH

machines_batch_source = FileSource(
    name="washing_batch_source",
    path="/data/offline/machines_batch_features", 
    timestamp_field='timestamp',
    description="Historical washing machine features stored in partitioned Parquet files"
)

### STREAMING

machines_stream_backing_source = FileSource(
    name='machines_stream_backing_source',
    path='/data/offline/machines_stream_source/streaming_feature_backfill.parquet',
    timestamp_field='timestamp'
)

machines_stream_source = PushSource(
    name="washing_stream_push",
    batch_source=machines_stream_backing_source,
    description="Push source for real-time features from Redpanda"
)