"""
Create offline storage files/directories required by the Feast feature store.

Paths created
─────────────
  1. /offline/machines_stream_source/streaming_feature_backfill.parquet
       → Single .parquet file (schema-only, no rows).
         Feast uses this as the historical backing store for the PushSource.
         Columns mirror `machine_streaming_features` FeatureView exactly.

  2. /offline/machines_batch_features/
       → Directory only. Feast / PySpark batch pipeline will write partitioned
         .parquet files inside automatically.
         An empty schema-only file is also placed here so the FileSource can
         be validated by Feast before the first batch run.

Entity column required by Feast
────────────────────────────────
  Machine_ID  (Int64) — the shared entity key for both FeatureViews.

Usage
─────
  pip install pandas pyarrow
  python create_feast_parquet_files.py
"""

import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────

STREAM_PARQUET_PATH = "/offline/machines_stream_source/streaming_feature_backfill.parquet"
BATCH_FEATURES_DIR  = "/offline/machines_batch_features/"
BATCH_INIT_FILE     = os.path.join(BATCH_FEATURES_DIR, "_init_schema.parquet")


# ──────────────────────────────────────────────────────────────────────────────
# SCHEMAS  (must match FeatureView definitions + Feast mandatory columns)
# ──────────────────────────────────────────────────────────────────────────────

# Feast expects these two timestamp columns in every historical parquet:
#   event_timestamp  – when the event occurred
#   created          – when the row was written (optional but recommended)

STREAM_SCHEMA = pa.schema([
    # ── Feast / entity columns ────────────────────────────────────────────────
    pa.field("Machine_ID",                      pa.int64()),
    pa.field("event_timestamp",                 pa.timestamp("us", tz="UTC")),
    pa.field("created",                         pa.timestamp("us", tz="UTC")),

    # ── Raw sensor readings ───────────────────────────────────────────────────
    pa.field("Cycle_Phase_ID",                  pa.int64()),
    pa.field("Current_L1",                      pa.float32()),
    pa.field("Current_L2",                      pa.float32()),
    pa.field("Current_L3",                      pa.float32()),
    pa.field("Voltage_L_L",                     pa.float32()),
    pa.field("Water_Temp_C",                    pa.float32()),
    pa.field("Motor_RPM",                       pa.float32()),
    pa.field("Water_Flow_L_min",                pa.float32()),
    pa.field("Vibration_mm_s",                  pa.float32()),
    pa.field("Water_Pressure_Bar",              pa.float32()),

    # ── Streaming pipeline features (rolling windows) ─────────────────────────
    pa.field("Current_Imbalance_Ratio",         pa.float32()),
    pa.field("Vibration_RollingMax_10min",      pa.float32()),
    pa.field("Current_Imbalance_RollingMean_5min", pa.float32()),
])

BATCH_SCHEMA = pa.schema([
    # ── Feast / entity columns ────────────────────────────────────────────────
    pa.field("Machine_ID",                      pa.int64()),
    pa.field("event_timestamp",                 pa.timestamp("us", tz="UTC")),
    pa.field("created",                         pa.timestamp("us", tz="UTC")),

    # ── Batch / aggregated features ───────────────────────────────────────────
    pa.field("Daily_Vibration_PeakMean_Ratio",  pa.float32()),
    pa.field("Weekly_Current_StdDev",           pa.float32()),
])


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    """Create directory (and any missing parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _write_empty_parquet(file_path: str, schema: pa.Schema) -> None:
    """Write a valid, zero-row parquet file with the given *schema*."""
    table = schema.empty_table()
    pq.write_table(table, file_path, compression="snappy")


def create_stream_parquet(path: str = STREAM_PARQUET_PATH) -> None:
    """
    Create the streaming backfill .parquet file if it does not already exist.

    The file will have the correct schema but zero rows.  The streaming
    pipeline (Quixstreams → Feast push) will append rows at runtime.
    """
    if os.path.isfile(path):
        print(f"[SKIP] Already exists : {path}")
        return

    _ensure_dir(os.path.dirname(path))
    _write_empty_parquet(path, STREAM_SCHEMA)
    print(f"[OK]   Stream parquet created : {path}")
    print(f"       Columns : {STREAM_SCHEMA.names}")


def create_batch_directory(
    dir_path:  str = BATCH_FEATURES_DIR,
    init_file: str = BATCH_INIT_FILE,
) -> None:
    """
    Create the batch features directory if it does not already exist.

    An empty schema-only parquet file (_init_schema.parquet) is also written
    so that `feast materialize` / `feast apply` can validate the FileSource
    before the first PySpark batch run populates the folder.
    """
    if os.path.isdir(dir_path):
        print(f"[SKIP] Directory already exists : {dir_path}")
    else:
        _ensure_dir(dir_path)
        print(f"[OK]   Batch directory created : {dir_path}")

    if os.path.isfile(init_file):
        print(f"[SKIP] Init file already exists : {init_file}")
        return

    _write_empty_parquet(init_file, BATCH_SCHEMA)
    print(f"[OK]   Batch init parquet created : {init_file}")
    print(f"       Columns : {BATCH_SCHEMA.names}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Feast offline storage initialisation")
    print(f" Run at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    print()

    create_stream_parquet()
    print()
    create_batch_directory()

    print()
    print("=" * 60)
    print(" Done. You can now run `feast apply` safely.")
    print("=" * 60)