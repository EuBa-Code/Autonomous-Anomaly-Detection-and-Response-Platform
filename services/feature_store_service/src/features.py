"""
Feature View Definitions for Feast Feature Store

Architecture
------------

  vibration_push_source ──► machine_streaming_features_10m   TTL: 10 min
  current_push_source   ──► machine_streaming_features_5min  TTL:  5 min
  machines_batch_source ──► machine_batch_features           TTL:  7 days

Why a dedicated PushSource per window?
  • Each FeatureView gets an independent TTL matched to its window length.
  • The streaming pipeline pushes directly to each endpoint with no stateful
    merge step; partial records are impossible by design.
  • Offline backfill paths are kept separate so point-in-time joins are clean.
  • Ownership is explicit: 10-min vibration and 5-min current are independent
    signals that happen to share the same entity key (Machine_ID).
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32

from entity import machine
from data_sources import (
    machines_batch_source,
    vibration_push_source,
    current_push_source,
)


# ── 10-min streaming feature view ─────────────────────────────────────────────
# Receives pushes from the 10-min sliding window in the streaming pipeline.
# before the next window fires.

machine_streaming_features_10m = FeatureView(
    name="machine_streaming_features_10m",
    entities=[machine],
    ttl=timedelta(minutes=15),          # window=10 min + 5 min grace
    schema=[
        # Rolling maximum of Vibration_mm_s over the last 10 minutes.
        # A sustained rise indicates mechanical wear or bearing failure.
        Field(name="Vibration_RollingMax_10min", dtype=Float32),
    ],
    source=vibration_push_source,
    description="10-min rolling vibration features pushed by the streaming pipeline",
)


# ── 5-min streaming feature view ──────────────────────────────────────────────
# Receives pushes from the 5-min sliding window in the streaming pipeline.

machine_streaming_features_5min = FeatureView(
    name="machine_streaming_features_5min",
    entities=[machine],
    ttl=timedelta(minutes=8),           # window=5 min + 3 min grace
    schema=[
        # Instantaneous 3-phase imbalance: (max-min) / mean across L1,L2,L3.
        # Computed per record before windowing; the latest value is carried through.
        Field(name="Current_Imbalance_Ratio",            dtype=Float32),
        # Rolling mean of the instantaneous ratio over the last 5 minutes.
        # Smooths out transient spikes to reveal a persistent imbalance trend.
        Field(name="Current_Imbalance_RollingMean_5min", dtype=Float32),
    ],
    source=current_push_source,
    description="5-min rolling current-imbalance features pushed by the streaming pipeline",
)


# ── Batch feature view ────────────────────────────────────────────────────────
# Written once a day by the PySpark batch pipeline.
# TTL of 7 days ensures weekly features are still served even if the batch
# pipeline is delayed by a day or two.

machine_batch_features = FeatureView(
    name="machine_batch_features",
    entities=[machine],
    ttl=timedelta(days=365 * 3), # Debugging cold start (bigger TTL)
    schema=[
        # max(Vibration) / mean(Vibration) per calendar day.
        # A high ratio means repeated or sustained shock events across hundreds
        # of cycles — a strong signal of mechanical deterioration.
        Field(name="Daily_Vibration_PeakMean_Ratio", dtype=Float32),
    ],
    source=machines_batch_source,
    description="Daily/weekly batch aggregation features written by PySpark",
)