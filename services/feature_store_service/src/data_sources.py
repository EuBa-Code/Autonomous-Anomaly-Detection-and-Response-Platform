"""
Data Sources for Feast Feature Store

This module defines both batch (offline) and streaming data sources
for the washing machine anomaly detection system.
"""

from feast import PushSource
from feast.infra.offline_stores.file_source import FileSource

#####
# Batch
#####

machines_batch = FileSource(
    name="washing_batch_source",
    path="/app/data/offline/machines_batch_features",
    timestamp_field='timestamp'
)

#####
# Streaming
#####

machines_stream_backing_source = FileSource(
    name='machines_stream_backing_source',
    path='/app/data/offline/push/washing_features_batch.parquet', 
    timestamp_field='timestamp',
    description='Batch source for historical washing machine features'
)

stream_source = PushSource(
    name='washing_stream_source',
    batch_source=machines_stream_backing_source,
    description='Push source for real-time washing machine features'
)