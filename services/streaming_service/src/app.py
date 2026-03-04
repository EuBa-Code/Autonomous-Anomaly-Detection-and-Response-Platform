"""
Streaming Transformation & Feature Ingestion Service

Pipeline overview
-----------------

  [Redpanda] telemetry-data
        │
        ├─ raw sink ──────────────────────────────────────► /data/entity_df   (Parquet)
        │                                                    (ground-truth for
        │                                                     batch PIT joins)
        │
        ├─ compute Current_Imbalance_Ratio (per record)
        │
        ├─ 10-min sliding window
        │     agg: Vibration_RollingMax_10min
        │     └──► POST /push → vibration_push_source ──► Feast (Redis + Parquet)
        │
        └─ 5-min sliding window
              agg: Current_Imbalance_RollingMean_5min
                   Current_Imbalance_Ratio (latest)
              └──► POST /push → current_push_source   ──► Feast (Redis + Parquet)

Design decisions
----------------
* Two push sources (one per window) replace the previous stateful-merge approach.
  Each window pushes only its own fields, so there are no partial/None records.
* The raw sink writes before any transformation to preserve the original signal.
* timestamp_extractor prefers the event timestamp embedded in the message over
  the Kafka broker timestamp to ensure correct windowing on late arrivals.
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from quixstreams import Application
from quixstreams.models import TimestampType
from quixstreams.sinks.community.file.local import LocalFileSink
from quixstreams.dataframe.windows import Latest, Mean, Max

from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("StreamingService")


# ── Timestamp utilities ───────────────────────────────────────────────────────

def _parse_iso8601(ts: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware datetime (UTC fallback)."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_epoch_ms(ts: str) -> int:
    """Convert an ISO-8601 string to epoch milliseconds."""
    return int(_parse_iso8601(ts).timestamp() * 1000)


def timestamp_extractor(
    value: Any,
    headers: Any,
    timestamp: float,
    timestamp_type: TimestampType,
) -> int:
    """
    QuixStreams timestamp extractor.

    Prefers the event timestamp embedded in the message payload so that
    sliding windows are aligned to when the event actually occurred, not
    when Kafka received it.  Falls back to the broker timestamp on any error.
    """
    try:
        if isinstance(value, dict) and "timestamp" in value:
            return _to_epoch_ms(value["timestamp"])
    except Exception as exc:
        logger.error(
            "Timestamp extractor error — falling back to broker timestamp: %s", exc
        )
    return int(timestamp)


# ── Feast push client ─────────────────────────────────────────────────────────

class FeastPusher:
    """Thin HTTP wrapper around the Feast feature server's /push endpoint.

    Two public helpers (push_vibration / push_current) map directly to the
    two push sources defined in data_sources.py.  Both delegate to the same
    private _push() method, which handles serialisation and error checking.
    """

    def __init__(self, base_url: str, push_to: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._push_to  = push_to
        self._session  = requests.Session()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def wait_until_ready(self, timeout_s: int = 120) -> None:
        """Block until the Feast server responds on /health or the timeout expires."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                resp = self._session.get(f"{self._base_url}/health", timeout=2)
                if resp.status_code == 200:
                    logger.info("Feast feature server is ready")
                    # Recycle the session so stale connections don't linger.
                    self._session.close()
                    self._session = requests.Session()
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("Feast feature server did not become ready in time")

    # ── Per-window push helpers ───────────────────────────────────────────────

    def push_vibration(self, record: dict) -> None:
        """Push 10-min vibration features to *vibration_push_source*."""
        payload = {
            "Machine_ID":                record["Machine_ID"],
            "timestamp":                 _parse_iso8601(record["latest_timestamp"]).isoformat(),
            "Vibration_RollingMax_10min": record["Vibration_RollingMax_10min"],
        }
        logger.info("Pushing vibration features: %s", payload)
        self._push(Config.PUSH_SOURCE_VIBRATION, payload)

    def push_current(self, record: dict) -> None:
        """Push 5-min current-imbalance features to *current_push_source*."""
        payload = {
            "Machine_ID":                        record["Machine_ID"],
            "timestamp":                         _parse_iso8601(record["latest_timestamp"]).isoformat(),
            "Current_Imbalance_Ratio":            record["Current_Imbalance_Ratio"],
            "Current_Imbalance_RollingMean_5min": record["Current_Imbalance_RollingMean_5min"],
        }
        logger.info("Pushing current features: %s", payload)
        self._push(Config.PUSH_SOURCE_CURRENT, payload)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _push(self, push_source_name: str, record: dict) -> None:
        """Serialise *record* and POST it to the Feast /push endpoint."""
        # Feast expects each field as a list (column-oriented dict / DataFrame).
        body = {
            "push_source_name": push_source_name,
            "to":               self._push_to,
            "df":               {k: [v] for k, v in record.items()},
        }
        resp = self._session.post(
            f"{self._base_url}/push",
            json=body,
            timeout=5,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 300:
            raise RuntimeError(
                f"Feast push failed [{push_source_name}]: "
                f"HTTP {resp.status_code} — {resp.text}"
            )


# ── Derived feature ───────────────────────────────────────────────────────────

def compute_current_imbalance_ratio(record: dict) -> float:
    """
    Compute the instantaneous 3-phase current imbalance scalar.

        ratio = (max(L1, L2, L3) - min(L1, L2, L3)) / mean(L1, L2, L3)

    Returns float('nan') when the mean is zero or any input is missing/invalid.
    This value is computed once per raw record and then fed into the 5-min
    sliding window aggregation.
    """
    try:
        c1 = float(record.get("Current_L1", 0.0))
        c2 = float(record.get("Current_L2", 0.0))
        c3 = float(record.get("Current_L3", 0.0))
        mean_val = (c1 + c2 + c3) / 3.0
        if mean_val == 0.0:
            logger.warning(
                "Mean current is zero for Machine_ID=%s — imbalance set to NaN",
                record.get("Machine_ID"),
            )
            return float("nan")
        return float((max(c1, c2, c3) - min(c1, c2, c3)) / mean_val)
    except Exception as exc:
        logger.error(
            "Failed to compute Current_Imbalance_Ratio for Machine_ID=%s: %s",
            record.get("Machine_ID"),
            exc,
        )
        return float("nan")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Starting streaming transformation service")

    feast = FeastPusher(
        base_url=Config.FEAST_SERVER_URL,
        push_to=Config.PUSH_TO,
    )
    feast.wait_until_ready()

    app = Application(
        broker_address=Config.KAFKA_SERVER,
        consumer_group=Config.TOPIC_TELEMETRY,
        auto_offset_reset=Config.AUTO_OFFSET_RESET,
        state_dir=Config.STATE_DIR,
    )

    # ── Input topic ───────────────────────────────────────────────────────────
    topic = app.topic(
        Config.TOPIC_TELEMETRY,
        value_deserializer="json",
        timestamp_extractor=timestamp_extractor,   # event-time windowing
    )

    # ── Raw sink (entity DataFrame) ───────────────────────────────────────────
    # Every raw message is persisted to Parquet before any transformation.
    # These files are the ground-truth input for the batch pipeline's
    # point-in-time join when building the historical feature tables.
    raw_sink = LocalFileSink(
        directory=Config.ENTITY_DF_DIR,
        format=Config.ENTITY_DF_FORMAT,
    )

    sdf = app.dataframe(topic)
    sdf.sink(raw_sink)

    # Derived per-record feature — computed here so it is available to the
    # 5-min window aggregation without duplicating the logic.
    sdf["Current_Imbalance_Ratio"] = sdf.apply(compute_current_imbalance_ratio)

    # ── 10-min sliding window → vibration features ────────────────────────────
    # grace_ms gives the watermark some slack for slightly late messages.
    (
        sdf
        .sliding_window(
            duration_ms=timedelta(minutes=10),
            grace_ms=timedelta(minutes=2),
        )
        .agg(
            Machine_ID=Latest("Machine_ID"),
            Vibration_RollingMax_10min=Max("Vibration_mm_s"),
            latest_timestamp=Latest("timestamp"),
        )
        .current()                          # emit on every new record
        .apply(feast.push_vibration)        # push directly — no merge needed
    )

    # ── 5-min sliding window → current-imbalance features ────────────────────
    (
        sdf
        .sliding_window(
            duration_ms=timedelta(minutes=5),
            grace_ms=timedelta(minutes=2),
        )
        .agg(
            Machine_ID=Latest("Machine_ID"),
            Current_Imbalance_Ratio=Latest("Current_Imbalance_Ratio"),
            Current_Imbalance_RollingMean_5min=Mean("Current_Imbalance_Ratio"),
            latest_timestamp=Latest("timestamp"),
        )
        .current()
        .apply(feast.push_current)
    )

    logger.info("Pipelines configured — starting app")
    app.run()

if __name__ == "__main__":
    main()