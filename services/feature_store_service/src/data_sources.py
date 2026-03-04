"""
Data Sources for Feast Feature Store

Two dedicated PushSources — one per streaming window — allow each FeatureView
to have its own TTL, its own offline backfill path, and its own push endpoint.
This avoids the need for a stateful merge in the streaming pipeline.

Layout
------
  Batch (FileSource)
    machines_batch_source          → machine_batch_features  (PySpark daily/weekly)

  Streaming (PushSource)
    vibration_push_source          → machine_streaming_features_10m
    current_push_source            → machine_streaming_features_5min
"""

from feast import PushSource
from feast.infra.offline_stores.file_source import FileSource


# ── Batch source ──────────────────────────────────────────────────────────────
# Written by the PySpark batch pipeline; partitioned Parquet directory.

machines_batch_source = FileSource(
    name="washing_batch_source",
    path="/data/offline/machines_batch_features",
    timestamp_field="timestamp",
    description="Historical washing machine features stored in partitioned Parquet files",
)


# ── Streaming backing sources (offline backfill for each push source) ─────────
# Each push source needs a FileSource as its offline backing store so that
# historical feature retrieval (point-in-time joins) still works.

_vibration_backing = FileSource(
    name="vibration_stream_backing_source",
    path="/data/offline/streaming_backfill/vibration",
    timestamp_field="timestamp",
    description="Offline backfill for the 10-min vibration rolling-max feature",
)

_current_backing = FileSource(
    name="current_stream_backing_source",
    path="/data/offline/streaming_backfill/current",
    timestamp_field="timestamp",
    description="Offline backfill for the 5-min current imbalance features",
)


# ── Push sources ──────────────────────────────────────────────────────────────
# The streaming pipeline calls POST /push with these names to write to the
# online store (Redis) and optionally to the offline backing Parquet.

vibration_push_source = PushSource(
    name="vibration_push_source",
    batch_source=_vibration_backing,
    description="Push source for 10-min rolling vibration features from Redpanda",
)

current_push_source = PushSource(
    name="current_push_source",
    batch_source=_current_backing,
    description="Push source for 5-min rolling current-imbalance features from Redpanda",
)