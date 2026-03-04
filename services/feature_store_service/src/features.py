"""
Feature View Definitions for Feast Feature Store

Architecture: two dedicated FeatureViews.

  ┌─────────────────────────────────┐      ┌──────────────────────────────────┐
  │  machine_streaming_features     │      │  machine_batch_features          │
  │  source : stream_source         │      │  source : machines_batch         │
  │  ttl    : 24 h                  │      │  ttl    : 7 days                 │
  │                                 │      │                                  │
  │  Raw sensor readings            │      │  Daily aggregations              │
  │  + rolling-window features      │      │  + weekly aggregations           │
  │    computed by the streaming    │      │    computed by the batch         │
  │    pipeline (Quixstreams)       │      │    pipeline (PySpark)            │
  └─────────────────────────────────┘      └──────────────────────────────────┘

Why split?
- The two pipelines have different refresh cadences: the streaming pipeline
  pushes every few seconds; the batch pipeline runs once a day/week.
- Using separate TTLs avoids evicting fresh streaming values just because the
  daily batch hasn't run yet, and vice-versa.
- It keeps feature ownership clear: streaming team owns
  machine_streaming_features; batch/data-engineering team owns
  machine_batch_features.
- Both views share the same entity (Machine_ID) so they can be joined at
  retrieval time with a single entity-row lookup.
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32 

from entity import machine
from data_sources import machines_batch_source, machines_stream_source


# ==============================================================================
# STREAMING FEATURE VIEW
# Source : stream_source (PushSource → backed by machines_batch for history)
# TTL    : 24 hours  — online store evicts stale entries after one day
#
# Contains:
#   • Raw sensor readings written by the ingestion / streaming service
#   • Rolling-window features pre-computed by the PySpark streaming pipeline:
#       - Vibration_RollingMax_10min        (10-min rolling max of Vibration_mm_s)
#       - Current_Imbalance_Ratio           (instantaneous 3-phase imbalance scalar)
#       - Current_Imbalance_RollingMean_5min (5-min rolling mean of the ratio above)
# ==============================================================================

machine_streaming_features_10m = FeatureView(
    name="machine_streaming_features_10m",
    entities=[machine],
    ttl=timedelta(minutes=10),
    schema=[
        # 10-minute rolling maximum of Vibration_mm_s (per Machine_ID).
        Field(name="Vibration_RollingMax_10min",        dtype=Float32),
    ],
    source=machines_stream_source,
)

machine_streaming_features_5min = FeatureView(
    name="machine_streaming_features_5min",
    entities=[machine],
    ttl=timedelta(minutes=5),
    schema=[
        # Intermediate derived scalar: (max(L1,L2,L3) - min(L1,L2,L3)) / mean(L1,L2,L3)
        Field(name="Current_Imbalance_Ratio",           dtype=Float32),
        Field(name="Current_Imbalance_RollingMean_5min", dtype=Float32),
    ],
    source=machines_stream_source,
)

# ==============================================================================
# BATCH FEATURE VIEW
# Source : machines_batch (FileSource → partitioned Parquet written by PySpark)
# TTL    : 7 days  — weekly features are valid for the entire week they cover;
#                    7 days ensures online store never serves a stale weekly value
#                    while still being tight enough to expire genuinely old data.
#
# Contains:
#   • Daily aggregation joined back to every row in that machine-day:
#       - Daily_Vibration_PeakMean_Ratio   max(Vibration) / mean(Vibration) per day
#   • Weekly aggregation joined back to every row in that machine-week:
#       - Weekly_Current_StdDev            stddev(Current_L1) per week
# ==============================================================================

machine_batch_features = FeatureView(
    name="machine_batch_features",
    entities=[machine],
    ttl=timedelta(days=7),
    schema=[
        # Daily peak-to-mean vibration ratio (per Machine_ID, per calendar day).
        # High ratio = repeated or sustained shock events across hundreds of
        # cycles → strong signal of mechanical deterioration.
        Field(name="Daily_Vibration_PeakMean_Ratio", dtype=Float32),
    ],
    source=machines_batch_source,
)


