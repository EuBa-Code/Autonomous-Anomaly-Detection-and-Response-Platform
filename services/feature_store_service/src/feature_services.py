"""
Feature Service Definitions for Feast Feature Store

A FeatureService groups feature views into a versioned contract consumed by a
specific ML model or application.  Callers request features by service name
rather than by individual view/field names, which decouples model code from
the underlying feature implementation.

Service v1 — machine_anomaly_service_v1
  Combines ALL features (streaming + batch) needed by the anomaly-detection model:
    • machine_streaming_features  – real-time sensor readings + rolling windows
    • machine_batch_features      – daily / weekly long-horizon aggregations

  Both views share the Machine_ID entity key, so a single entity-row lookup
  returns the full feature vector at inference time.
"""

from feast import FeatureService

from features import machine_streaming_features_10m, machine_streaming_features_5min, machine_batch_features


# ==============================================================================
# ANOMALY DETECTION SERVICE — v1
# ==============================================================================

machine_anomaly_service_v1 = FeatureService(
    name="machine_anomaly_service_v1",
    features=[
        machine_streaming_features_10m, 
        machine_streaming_features_5min,    # Raw sensors + rolling-window features
        machine_batch_features,        # Daily / weekly aggregation features
    ],
    description=(
        "Full feature vector for washing-machine anomaly detection model v1. "
        "Combines real-time rolling-window signals (streaming pipeline) with "
        "daily and weekly aggregation context (batch pipeline)."
    ),
)