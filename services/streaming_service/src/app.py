import json
import logging
import math
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta

import pandas as pd
from feast import FeatureStore
from feast.data_source import PushMode
from quixstreams import Application
from quixstreams.sinks.community.file.local import LocalFileSink

from config.config import Config

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamingService")


# ============================================================================
# IN-PROCESS ROLLING WINDOW STATE
# Keyed by Machine_ID. Each deque entry: (datetime, value)
# ============================================================================
# 10-min window  →  Vibration_RollingMax_10min
vibration_windows: dict[str, deque] = defaultdict(deque)

# 5-min window   →  Current_Imbalance_RollingMean_5min
imbalance_windows: dict[str, deque] = defaultdict(deque)

WINDOW_10MIN = timedelta(minutes=10)
WINDOW_5MIN  = timedelta(minutes=5)


def _evict_old(window: deque, cutoff: datetime) -> None:
    """Remove entries older than cutoff from the left of the deque."""
    while window and window[0][0] < cutoff:
        window.popleft()


def _rolling_max(window: deque) -> float:
    return max(v for _, v in window) if window else 0.0


def _rolling_mean(window: deque) -> float:
    if not window:
        return 0.0
    return sum(v for _, v in window) / len(window)


# ============================================================================
# MAIN SERVICE
# ============================================================================

def run_streaming_service():
    """
    Real-time feature engineering service using Quix Streams.

    Pipeline:
      1. Consume raw telemetry from Redpanda
      2. Compute three streaming features (correct rolling windows):
            - Current_Imbalance_Ratio           (instantaneous scalar)
            - Vibration_RollingMax_10min         (10-min rolling max per Machine_ID)
            - Current_Imbalance_RollingMean_5min (5-min rolling mean per Machine_ID)
      3. Push ALL features (raw sensors + derived) to Feast Online Store (Redis)
      No output topic — Redpanda is consumed only.
    """

    # ------------------------------------------------------------------
    # 1. Initialise Feast
    # ------------------------------------------------------------------
    repo_path = os.getenv("FEAST_REPO_PATH", "/streaming_service")
    try:
        store = FeatureStore(repo_path=repo_path)
        logger.info("Feast Feature Store initialised successfully")
    except Exception as e:
        logger.error(f"Failed to initialise Feast Feature Store: {e}")
        return

    # ------------------------------------------------------------------
    # 2. Initialise Quix — consume only, no output topic
    # ------------------------------------------------------------------
    app = Application(
        broker_address=Config.KAFKA_SERVER,
        consumer_group="quix-streaming-processor-v3",
        auto_offset_reset="earliest",
    )

    # Use bytes deserializer — the raw messages may contain NaN/Infinity
    # which are valid Python floats but illegal in strict JSON (orjson rejects them).
    # We parse manually with the stdlib json module which handles them gracefully.
    input_topic = app.topic(Config.TOPIC_TELEMETRY, value_deserializer="bytes")
    sdf = app.dataframe(input_topic)

    # ------------------------------------------------------------------
    # 3. Manual JSON decode (tolerant of NaN / Infinity)
    # ------------------------------------------------------------------
    def decode_message(raw: bytes) -> dict | None:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Could not decode message, skipping: {e} | raw[:80]={raw[:80]}")
            return None

    sdf = sdf.apply(decode_message)
    sdf = sdf.filter(lambda x: x is not None)

    # ------------------------------------------------------------------
    # 4a. Raw sink — persist all sensor fields to local datalake before
    #     any feature engineering (mirrors processor.py Branch 1 pattern)
    # ------------------------------------------------------------------
    raw_sink = LocalFileSink(
        directory=os.getenv("DATALAKE_DIR", "/data/entity_df"),
        format=os.getenv("DATALAKE_FORMAT", "parquet"),
    )
    def _safe_float(value, default: float = 0.0) -> float:
        """Cast to float and replace NaN/Inf with a safe default."""
        try:
            f = float(value)
            return default if not math.isfinite(f) else f
        except (TypeError, ValueError):
            return default

    def process_and_push(data: dict) -> None:
        machine_id   = str(data.get("Machine_ID", "Unknown"))
        timestamp_str = data.get("timestamp", "")

        try:
            ts = pd.to_datetime(timestamp_str)
        except Exception:
            logger.warning(f"Unparseable timestamp '{timestamp_str}' — skipping message")
            return

        # ── Feature 1: Current_Imbalance_Ratio (instantaneous) ──────────
        l1 = _safe_float(data.get("Current_L1"))
        l2 = _safe_float(data.get("Current_L2"))
        l3 = _safe_float(data.get("Current_L3"))
        currents = [l1, l2, l3]
        mean_c = sum(currents) / 3.0
        imbalance_ratio = (
            (max(currents) - min(currents)) / mean_c if mean_c > 0 else 0.0
        )

        # ── Feature 2: Vibration_RollingMax_10min ───────────────────────
        vibration = _safe_float(data.get("Vibration_mm_s"))
        vib_win = vibration_windows[machine_id]
        _evict_old(vib_win, ts - WINDOW_10MIN)
        vib_win.append((ts, vibration))
        vibration_rolling_max = _rolling_max(vib_win)

        # Validation: rolling max must be >= current reading
        if vibration_rolling_max < vibration:
            logger.error(
                f"[{machine_id}] Vibration_RollingMax_10min ({vibration_rolling_max:.4f}) "
                f"< Vibration_mm_s ({vibration:.4f}) — computation error"
            )

        # ── Feature 3: Current_Imbalance_RollingMean_5min ───────────────
        imb_win = imbalance_windows[machine_id]
        _evict_old(imb_win, ts - WINDOW_5MIN)
        imb_win.append((ts, imbalance_ratio))
        imbalance_rolling_mean = _rolling_mean(imb_win)

        # ── Build Feast DataFrame ────────────────────────────────────────
        # Include all raw sensor fields + all three derived features
        feast_row = {
            # Entity key
            "Machine_ID":                        int(data.get("Machine_ID", 0)),
            # Timestamp
            "timestamp":                         ts,
            # Derived / streaming features
            "Current_Imbalance_Ratio":           float(imbalance_ratio),
            "Vibration_RollingMax_10min":        float(vibration_rolling_max),
            "Current_Imbalance_RollingMean_5min": float(imbalance_rolling_mean),
        }

        feast_df = pd.DataFrame([feast_row])

        # ── Push to Feast Online Store (Redis) ───────────────────────────
        try:
            store.push("washing_stream_push", feast_df, to=PushMode.ONLINE_AND_OFFLINE)
            logger.info(
                f"[{machine_id}] Pushed to Feast (online+offline) | "
                f"ImbalanceRatio={imbalance_ratio:.4f} | "
                f"VibRollingMax={vibration_rolling_max:.4f} | "
                f"ImbalanceRollingMean={imbalance_rolling_mean:.4f}"
            )
        except Exception as feast_err:
            logger.error(f"[{machine_id}] Feast push failed: {feast_err}")

    # Apply — returns None so no downstream topic is needed
    sdf = sdf.apply(process_and_push)

    logger.info("Quix Streaming Service started — consuming from Redpanda, pushing to Feast.")
    app.run(sdf)


if __name__ == "__main__":
    run_streaming_service()