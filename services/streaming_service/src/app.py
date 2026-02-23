import logging
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta

import pandas as pd
from feast import FeatureStore
from quixstreams import Application

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

    input_topic = app.topic(Config.TOPIC_TELEMETRY, value_deserializer="json")
    sdf = app.dataframe(input_topic)

    # ------------------------------------------------------------------
    # 3. Feature engineering + Feast push
    # ------------------------------------------------------------------
    def process_and_push(data: dict) -> None:
        machine_id   = str(data.get("Machine_ID", "Unknown"))
        timestamp_str = data.get("timestamp", "")

        try:
            ts = pd.to_datetime(timestamp_str)
        except Exception:
            logger.warning(f"Unparseable timestamp '{timestamp_str}' — skipping message")
            return

        # ── Feature 1: Current_Imbalance_Ratio (instantaneous) ──────────
        l1 = float(data.get("Current_L1", 0.0))
        l2 = float(data.get("Current_L2", 0.0))
        l3 = float(data.get("Current_L3", 0.0))
        currents = [l1, l2, l3]
        mean_c = sum(currents) / 3.0
        imbalance_ratio = (
            (max(currents) - min(currents)) / mean_c if mean_c > 0 else 0.0
        )

        # ── Feature 2: Vibration_RollingMax_10min ───────────────────────
        vibration = float(data.get("Vibration_mm_s", 0.0))
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
            # Raw sensor readings (match schema in features.py)
            "Cycle_Phase_ID":                    int(data.get("Cycle_Phase_ID", 0)),
            "Current_L1":                        float(l1),
            "Current_L2":                        float(l2),
            "Current_L3":                        float(l3),
            "Voltage_L_L":                       float(data.get("Voltage_L_L", 0.0)),
            "Water_Temp_C":                      float(data.get("Water_Temp_C", 0.0)),
            "Motor_RPM":                         float(data.get("Motor_RPM", 0.0)),
            "Water_Flow_L_min":                  float(data.get("Water_Flow_L_min", 0.0)),
            "Vibration_mm_s":                    float(vibration),
            "Water_Pressure_Bar":                float(data.get("Water_Pressure_Bar", 0.0)),
            # Derived / streaming features
            "Current_Imbalance_Ratio":           float(imbalance_ratio),
            "Vibration_RollingMax_10min":        float(vibration_rolling_max),
            "Current_Imbalance_RollingMean_5min": float(imbalance_rolling_mean),
        }

        feast_df = pd.DataFrame([feast_row])

        # ── Push to Feast Online Store (Redis) ───────────────────────────
        try:
            store.push("washing_stream_push", feast_df)
            logger.info(
                f"[{machine_id}] Pushed to Feast | "
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