"""
Batch Feature Pipeline — Washing Machine Anomaly Detection
==========================================================

Reads processed sensor data from the datalake and computes ONE row per machine
containing the LAST complete daily window only.

Output schema (matches BATCH_SCHEMA / machine_batch_features FeatureView):
  Machine_ID                    Int64
  timestamp                     Timestamp UTC   ← latest event_timestamp in the window
  Daily_Vibration_PeakMean_Ratio Float32

Why "last window only"?
  • The offline store is materialised daily; older windows are already stored.
  • Writing only the latest window keeps the append fast and avoids re-computing
    history that Feast already has.
  • The `latest_window_per_entity` pattern (from job.py) picks exactly that window.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    datalake_dir:    str   # Source: processed industrial washer features (full history)
    offline_dir:     str   # Destination: Feast offline store directory
    spark_partitions: int  # Output parquet file count


def load_settings() -> Settings:
    return Settings(
        datalake_dir=os.getenv(
            "HISTORICAL_DIR",
            "/app/data/processed_datasets/industrial_washer_normal_features",
        ),
        offline_dir=os.getenv(
            "OFFLINE_DIR",
            "/app/data/offline/machines_batch_features",
        ),
        spark_partitions=int(os.getenv("SPARK_PARTITIONS", "8")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read_parquet(spark: SparkSession, path: Path) -> DataFrame:
    """Read parquet file(s) — raises immediately if path is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")
    logger.info(f"Reading parquet from: {path}")
    df = spark.read.parquet(str(path))
    logger.info(f"  → {df.count()} rows loaded")
    return df


def latest_window_per_entity(df: DataFrame, entity_col: str) -> DataFrame:
    """
    Given a DataFrame that already has a 'window' struct column (from F.window()),
    return only the row belonging to the MOST RECENT window for each entity value.

    Mirrors the pattern in job.py:
        w = Window.partitionBy(entity_col).orderBy(F.col("window.end").desc())
        row_number == 1  →  latest window only
    """
    w = Window.partitionBy(entity_col).orderBy(F.col("window.end").desc())
    return (
        df.withColumn("_rn", F.row_number().over(w))
          .filter(F.col("_rn") == 1)
          .drop("_rn")
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def read_inputs(spark: SparkSession, datalake_dir: str) -> DataFrame:
    """
    Read the full processed feature dataset from the datalake.

    Expected columns (subset used here):
      Machine_ID      Int64
      timestamp       Timestamp UTC
      Vibration_mm_s  Float32   ← the only sensor column needed for this feature
    """
    return _read_parquet(spark, Path(datalake_dir))


def compute_last_daily_window(df: DataFrame, timestamp_col: str = "timestamp") -> DataFrame:
    """
    Compute Daily_Vibration_PeakMean_Ratio for the LAST complete daily window
    per machine.

    Steps
    ─────
    1. Ensure timestamp column is properly cast.
    2. Group by (Machine_ID, 1-day tumbling window) and aggregate:
         Daily_Vibration_PeakMean_Ratio = max(Vibration_mm_s) / mean(Vibration_mm_s)
         timestamp = max(timestamp) in that window   ← Feast needs a timestamp field
    3. Call latest_window_per_entity() to keep ONLY the most recent daily window
       per Machine_ID.
    4. Drop the 'window' struct — it is an internal Spark column not in BATCH_SCHEMA.

    Output columns:
      Machine_ID                     Int64
      timestamp                      Timestamp UTC
      Daily_Vibration_PeakMean_Ratio Float32
    """
    logger.info("Step 1 — casting timestamp column...")
    df = df.withColumn(timestamp_col, F.col(timestamp_col).cast("timestamp"))

    # ── Step 2: daily tumbling window aggregation ─────────────────────────────
    logger.info("Step 2 — aggregating into 1-day tumbling windows per Machine_ID...")
    daily_windows = (
        df.groupBy(
            F.col("Machine_ID"),
            F.window(timestamp_col, "1 day").alias("window"),   # struct: {start, end}
        )
        .agg(
            # Peak-to-mean vibration ratio: high value → impulsive/spiky behaviour
            (F.max("Vibration_mm_s") / F.mean("Vibration_mm_s"))
                .cast("float")
                .alias("Daily_Vibration_PeakMean_Ratio"),

            # Keep the latest raw timestamp inside this window so Feast can use it
            # as the event_timestamp for point-in-time joins.
            F.max(timestamp_col).alias(timestamp_col),
        )
    )

    # ── Step 3: keep only the LAST window per machine ─────────────────────────
    logger.info("Step 3 — selecting latest daily window per Machine_ID...")
    latest = latest_window_per_entity(daily_windows, "Machine_ID")

    # ── Step 4: clean up — drop the Spark window struct ──────────────────────
    result = latest.select(
        F.col("Machine_ID"),
        F.col(timestamp_col),
        F.col("Daily_Vibration_PeakMean_Ratio"),
    )

    logger.info("  → Daily batch features computed for last window only")
    return result


def write_offline(df: DataFrame, out_path: Path, partitions: int) -> None:
    """
    Write feature rows to the Feast offline store (partitioned parquet directory).

    Write mode is 'append' so that previous days' windows are preserved in the
    directory and Feast can serve point-in-time lookups over the full history.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to offline store: {out_path}  (partitions={partitions})")
    (
        df.repartition(partitions)
          .write
          .mode("append")            # append: daily incremental, no history overwrite
          .parquet(str(out_path))
    )
    logger.info("  → Written successfully")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("BATCH FEATURE PIPELINE  —  last daily window per machine")
    logger.info("=" * 70)

    s = load_settings()

    # 1. Spark session
    spark = (
        SparkSession.builder
        .appName("batch-feature-pipeline-washing-machines")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(max(8, s.spark_partitions * 2)))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session ready")

    try:
        # 2. Read full datalake (all machines, all time)
        df = read_inputs(spark, s.datalake_dir)

        # 3. Compute feature for the last daily window only
        batch_features = compute_last_daily_window(df, timestamp_col="timestamp")

        # 4. Preview
        logger.info("Sample output:")
        batch_features.show(10, truncate=False)

        # 5. Append to offline store
        write_offline(batch_features, Path(s.offline_dir), s.spark_partitions)

        # 6. Summary
        end_ts = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Offline store : {s.offline_dir}")
        logger.info(f"  Feature       : Daily_Vibration_PeakMean_Ratio")
        logger.info(f"  Suggested end-date for feast materialize-incremental: {end_ts}")
        logger.info("=" * 70)

    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        raise

    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()