"""
Initialise the offline storage files and directories required by the Feast
feature store.

Paths created
─────────────
  1. /data/offline/streaming_backfill/vibration/_init_schema.parquet
       Schema-only (zero rows) backing file for `vibration_push_source`.
       Feast uses this path as the historical store for the 10-min
       vibration FeatureView (machine_streaming_features_10m).

  2. /data/offline/streaming_backfill/current/_init_schema.parquet
       Schema-only (zero rows) backing file for `current_push_source`.
       Feast uses this path as the historical store for the 5-min
       current-imbalance FeatureView (machine_streaming_features_5min).

  3. /data/offline/machines_batch_features/_init_schema.parquet
       Schema-only (zero rows) seed file for the batch FileSource.
       The PySpark batch pipeline will write partitioned Parquet files
       alongside this file; Feast can validate the source before the
       first batch run.

All paths mirror the FileSource.path values declared in data_sources.py.
Schemas mirror the Field definitions in features.py exactly.

Usage
─────
  pip install pyarrow
  python create_feast_parquet_files.py
"""

import os
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq


# ── Paths (must match data_sources.py exactly) ────────────────────────────────

# Use relative paths so it stays inside your project folder
BASE_DATA_DIR = "data/offline" 

VIBRATION_BACKFILL_DIR = os.path.join(BASE_DATA_DIR, "streaming_backfill/vibration")
CURRENT_BACKFILL_DIR   = os.path.join(BASE_DATA_DIR, "streaming_backfill/current")
BATCH_FEATURES_DIR      = os.path.join(BASE_DATA_DIR, "machines_batch_features")

# Necessary to do materialization (for the first time)
DATALAKE = "data\processed_datasets\machines_batch_features"

# ── Schemas (must match features.py Field definitions + entity key) ───────────

# machine_streaming_features_10m  →  vibration_push_source
VIBRATION_SCHEMA = pa.schema([
    pa.field("Machine_ID",               pa.int64()),
    pa.field("timestamp",                pa.timestamp("us", tz="UTC")),
    pa.field("Vibration_RollingMax_10min", pa.float32()),
])

# machine_streaming_features_5min  →  current_push_source
CURRENT_SCHEMA = pa.schema([
    pa.field("Machine_ID",                        pa.int64()),
    pa.field("timestamp",                         pa.timestamp("us", tz="UTC")),
    pa.field("Current_Imbalance_Ratio",           pa.float32()),
    pa.field("Current_Imbalance_RollingMean_5min", pa.float32()),
])

# machine_batch_features  →  machines_batch_source (FileSource)
BATCH_SCHEMA = pa.schema([
    pa.field("Machine_ID",                    pa.int64()),
    pa.field("timestamp",                     pa.timestamp("us", tz="UTC")),
    pa.field("Daily_Vibration_PeakMean_Ratio", pa.float32()),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_init_parquet(directory: str, schema: pa.Schema, personalized: str = 'init.parquet', batch: bool = False) -> None:
    """
    Create *directory* (and any missing parents) and write a zero-row Parquet
    file named '_init_schema.parquet' with the given *schema*.

    Skips silently if the init file already exists so the script is safe to
    re-run without overwriting live data written by the pipelines.
    """
    
    init_path = os.path.join(directory, personalized)

    if os.path.isfile(init_path):
        print(f"[SKIP] Already exists : {init_path}")
        return

    os.makedirs(directory, exist_ok=True)
    pq.write_table(schema.empty_table(), init_path, compression="snappy")

    print(f"[OK]   Created : {init_path}")
    print(f"       Columns : {schema.names}")

def single_materialization() -> None:
    """
    Read ALL partitioned Parquet files written by PySpark into DATALAKE,
    keep only the most-recent row per Machine_ID (3 machines → 3 rows),
    and write the result into the Feast offline-store path
    (BATCH_FEATURES_DIR/materialization_seed.parquet).

    Only the three columns declared in BATCH_SCHEMA are retained:
        Machine_ID | timestamp | Daily_Vibration_PeakMean_Ratio
    """
    import pyarrow.dataset as ds

    print("── Single materialisation seed ──")

    # ── 1. Load every part-*.parquet produced by the Spark job ───────────────
    print(f"   Reading dataset from : {DATALAKE}")
    dataset = ds.dataset(DATALAKE, format="parquet")

    # Project only the three columns we care about (saves memory on large sets)
    table = dataset.to_table(
        columns=["Machine_ID", "timestamp", "Daily_Vibration_PeakMean_Ratio"]
    )
    print(f"   Total rows loaded    : {len(table):,}")

    # ── 2. Sort descending by timestamp, then keep first hit per machine ──────
    #    After a descending sort the first occurrence of each Machine_ID is its
    #    most-recent record — exactly one row per machine.
    sorted_table = table.sort_by([("timestamp", "descending")])

    df = sorted_table.to_pandas()
    last_records = (
        df.drop_duplicates(subset=["Machine_ID"], keep="first")
          .reset_index(drop=True)
    )

    print(f"   Machines found       : {sorted(last_records['Machine_ID'].tolist())}")
    print(f"   Rows kept (1/machine): {len(last_records)}")
    print(last_records.to_string(index=False))

    # ── 3. Cast back to the canonical Feast schema & write to offline store ───
    result_table = pa.Table.from_pandas(
        last_records,
        schema=BATCH_SCHEMA,
        preserve_index=False,
    )

    output_path = os.path.join(BATCH_FEATURES_DIR, "_init_schema.parquet")

    if not os.path.isfile(output_path):
        raise FileNotFoundError(
            f"[ERROR] {output_path} not found. "
            "Run _write_init_parquet() first to create the schema file."
        )

    pq.write_table(result_table, output_path, compression="snappy")

    print(f"\n[OK]   Records written into : {output_path}")
    print(f"       Columns              : {BATCH_SCHEMA.names}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Feast offline storage initialisation")
    print(f" Run at : {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    print()

    print("── Streaming backfill: vibration (10-min window) ──")
    _write_init_parquet(VIBRATION_BACKFILL_DIR, VIBRATION_SCHEMA, 'vibration_backfill.parquet')
    print()

    print("── Streaming backfill: current imbalance (5-min window) ──")
    _write_init_parquet(CURRENT_BACKFILL_DIR, CURRENT_SCHEMA, 'current_backfill.parquet')
    print()

    print("── Batch features (PySpark daily/weekly) ──")
    _write_init_parquet(BATCH_FEATURES_DIR, BATCH_SCHEMA, "_init_schema.parquet")
    single_materialization()  
    print()

    print("=" * 60)
    print(" Done. You can now run `feast apply` safely.")
    print("=" * 60)