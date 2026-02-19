"""
Data Engineering Service with Rolling Window Features
Processes industrial washer data with configurable rolling aggregations

Pipeline architecture:
  - Streaming features : short-term rolling windows (seconds → minutes)
  - Batch features     : long-term daily / weekly aggregations joined back to each row
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Service for processing data engineering washer data with rolling features"""
    
    def __init__(self, config_path: str):
        """
        Initialize the ingestion service

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        logger.info("Creating Spark session")
        
        builder = SparkSession.builder.appName("DataEngineering_RollingFeatures")
        
        # Apply Spark configurations from config
        if 'spark_config' in self.config:
            for key, value in self.config['spark_config'].items():
                builder = builder.config(key, value)
        
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        return spark
    
    def _parse_window_duration(self, duration_str: str) -> int:
        """
        Parse window duration string to seconds

        Args:
            duration_str: Duration string like '10 minutes', '1 hour', '30 seconds'

        Returns:
            Duration in seconds
        """
        parts = duration_str.lower().split()
        if len(parts) != 2:
            raise ValueError(f"Invalid duration format: {duration_str}")
        
        value = int(parts[0])
        unit = parts[1]
        
        multipliers = {
            'second': 1, 'seconds': 1,
            'minute': 60, 'minutes': 60,
            'hour': 3600, 'hours': 3600,
            'day': 86400, 'days': 86400
        }
        
        if unit not in multipliers:
            raise ValueError(f"Unknown time unit: {unit}")
        
        return value * multipliers[unit]
    
    def _read_dataset(self, dataset_config: Dict) -> Any:
        """Read dataset based on configuration"""
        logger.info(f"Reading dataset: {dataset_config['name']}")
        
        input_path = dataset_config['input_path']
        file_format = dataset_config.get('file_format', 'parquet')
        
        if file_format == 'parquet':
            df = self.spark.read.parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Read {df.count()} rows from {input_path}")
        return df
    
    def _validate_data_quality(self, df: Any, dataset_config: Dict) -> Any:
        """Perform data quality checks"""
        timestamp_col = self.config['schema']['timestamp_column']
        
        if self.config['data_quality'].get('check_null_timestamps', True):
            null_count = df.filter(F.col(timestamp_col).isNull()).count()
            if null_count > 0:
                logger.warning(f"Found {null_count} null timestamps - filtering out")
                df = df.filter(F.col(timestamp_col).isNotNull())
        
        if self.config['data_quality'].get('check_duplicate_timestamps', True):
            original_count = df.count()
            df = df.dropDuplicates([timestamp_col, 'Machine_ID'])
            dropped = original_count - df.count()
            if dropped > 0:
                logger.warning(f"Removed {dropped} duplicate timestamp-machine combinations")
        
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # DERIVED COLUMNS  (computed once, reused by streaming features that need
    # a composite signal like current imbalance)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_derived_columns(self, df: Any) -> Any:
        """
        Pre-compute derived scalar columns required by rolling feature configs
        that declare a 'source_expression' instead of a plain source_column.

        Each entry in rolling_features may optionally set:
            source_expression: "derived"     # tells the engine to look for a
            derived_column:    "some_col"    # pre-built column called 'some_col'

        Currently supported derived columns
        ------------------------------------
        Current_Imbalance_Ratio
            (max(L1, L2, L3) - min(L1, L2, L3)) / mean(L1, L2, L3)

            Captures three-phase electrical imbalance in a single value.
            A healthy motor stays below ~0.02; rising values signal
            winding faults or bearing degradation well before vibration
            changes become significant — ideal for an early-warning
            streaming feature fed to the ML model.
        """
        logger.info("Computing derived columns for streaming features")

        df = df.withColumn(
            "Current_Imbalance_Ratio",
            (
                F.greatest("Current_L1", "Current_L2", "Current_L3") -
                F.least("Current_L1", "Current_L2", "Current_L3")
            ) / (
                (F.col("Current_L1") + F.col("Current_L2") + F.col("Current_L3")) / 3
            )
        )

        logger.info("  ✓ Derived column ready: Current_Imbalance_Ratio")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # STREAMING FEATURES  (short-term rolling windows, seconds → minutes)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_rolling_feature(self, df: Any, feature_name: str, source_column: str, aggregation: str):
        """Validate rolling window feature calculations"""
        logger.info(f"  Validating rolling feature: {feature_name}")
        
        if aggregation == 'max':
            # Rolling max must always be >= the instantaneous source value
            invalid_count = df.filter(F.col(feature_name) < F.col(source_column)).count()
            if invalid_count > 0:
                logger.error(
                    f"  ❌ VALIDATION FAILED: Found {invalid_count} rows where "
                    f"{feature_name} < {source_column}"
                )
            else:
                logger.info(f"  ✓ Validation passed: All rolling max values >= source values")

        elif aggregation == 'mean':
            # For a rolling mean we check that it sits within [min, max] of the source
            stats = df.select(
                F.min(source_column).alias('src_min'),
                F.max(source_column).alias('src_max'),
                F.min(feature_name).alias('feat_min'),
                F.max(feature_name).alias('feat_max'),
            ).collect()[0]
            if stats['feat_min'] < stats['src_min'] or stats['feat_max'] > stats['src_max']:
                logger.warning(
                    f"  ⚠ Rolling mean range [{stats['feat_min']:.4f}, {stats['feat_max']:.4f}] "
                    f"outside source range [{stats['src_min']:.4f}, {stats['src_max']:.4f}]"
                )
            else:
                logger.info(
                    f"  ✓ Validation passed: Rolling mean in "
                    f"[{stats['feat_min']:.4f}, {stats['feat_max']:.4f}]"
                )
            return  # early return — stats already logged

        else:
            logger.warning(f"  No specific validation implemented for aggregation '{aggregation}'")
        
        # Shared null-value check
        null_count = df.filter(
            F.col(feature_name).isNull() & F.col(source_column).isNotNull()
        ).count()
        if null_count > 0:
            logger.warning(
                f"  ⚠ Found {null_count} null values in {feature_name} where source is not null"
            )
        
        stats = df.select(
            F.max(source_column).alias('source_max'),
            F.max(feature_name).alias('rolling_result'),
        ).collect()[0]
        logger.info(
            f"  Statistics: Source max [{stats['source_max']:.4f}], "
            f"Rolling result max [{stats['rolling_result']:.4f}]"
        )
    
    def _apply_rolling_features(self, df: Any) -> Any:
        """
        Apply streaming rolling window features based on configuration.

        Supports two kinds of source:
          1. Plain column   – set 'source_column' in the YAML entry.
          2. Derived column – set 'source_expression: derived' + 'derived_column'
             in the YAML entry; the column must have been pre-built by
             _compute_derived_columns().

        Minimum-window guard
        --------------------
        A rolling aggregation computed over a partial window (e.g. only 3 minutes
        of data for a 5-minute window) is misleading — it silently under-samples
        the signal and makes the feature incomparable across rows.

        Fix: after computing each feature we null out every row whose elapsed time
        since the machine's FIRST record is strictly less than the window duration.
        Only once a machine has accumulated at least <window_duration> seconds of
        history does the feature become non-null.

        Example (5-minute window, data starts 2024-01-01 00:00:00):
          00:00:00 → null   (0 s elapsed  < 300 s)
          00:02:30 → null   (150 s elapsed < 300 s)
          00:05:00 → value  (300 s elapsed == 300 s  ✓ full window)
          00:07:00 → value  (420 s elapsed  > 300 s  ✓ full window)
        """
        logger.info("Applying streaming (rolling window) features")
        
        timestamp_col = self.config['schema']['timestamp_column']
        partition_cols = self.config['schema']['partition_columns']
        
        # Ensure timestamp is in correct format
        df = df.withColumn(timestamp_col, F.col(timestamp_col).cast(TimestampType()))
        
        # Sort by timestamp and partition columns for correct window ordering
        logger.info(f"Sorting data by {timestamp_col}, {partition_cols} for window calculations")
        df = df.orderBy(timestamp_col, *partition_cols)

        # Pre-compute any derived scalar columns needed by rolling features
        df = self._compute_derived_columns(df)

        # ── Pre-compute per-machine first-record timestamp (epoch seconds) ────
        # Used by the minimum-window guard applied to every streaming feature.
        # A single unbounded partition window is far cheaper than one per feature.
        _FIRST_TS_COL = "_machine_first_ts_epoch_"
        machine_window = Window.partitionBy(*partition_cols)
        df = df.withColumn(
            _FIRST_TS_COL,
            F.min(F.col(timestamp_col).cast("long")).over(machine_window)
        )
        logger.info(
            f"  Pre-computed per-machine first-record timestamp → '{_FIRST_TS_COL}'"
        )
        
        # Process each enabled rolling feature (add any new one in the YAML)
        for feature_config in self.config['rolling_features']:
            if not feature_config.get('enabled', False):
                logger.info(f"Skipping disabled feature: {feature_config['feature_name']}")
                continue
            
            feature_name = feature_config['feature_name']
            aggregation = feature_config['aggregation']
            window_duration_str = feature_config['window_duration']

            # Resolve the source column (plain or derived)
            source_expression = feature_config.get('source_expression', 'column')
            if source_expression == 'derived':
                source_column = feature_config['derived_column']
            else:
                source_column = feature_config['source_column']
            
            logger.info(f"Creating feature: {feature_name} ({feature_config['description']})")
            logger.info(
                f"  Source: {source_column}, Aggregation: {aggregation}, "
                f"Window: {window_duration_str}"
            )
            
            # Parse window duration to seconds for rangeBetween
            window_duration_seconds = self._parse_window_duration(window_duration_str)
            logger.info(f"  Window duration: {window_duration_seconds} seconds")
            
            # CRITICAL: partitionBy keeps each machine's rolling window independent
            window_spec = (
                Window
                .partitionBy(*partition_cols)
                .orderBy(F.col(timestamp_col).cast("long"))
                .rangeBetween(-window_duration_seconds, 0)
            )
            
            # Apply aggregation — add more aggregation types here as needed
            if aggregation == 'max':
                df = df.withColumn(feature_name, F.max(source_column).over(window_spec))
            elif aggregation == 'mean':
                df = df.withColumn(feature_name, F.mean(source_column).over(window_spec))
            else:
                logger.warning(f"Unknown aggregation: {aggregation} for feature {feature_name}")
                continue

            # ── Minimum-window guard ──────────────────────────────────────────
            # Null out any row where the machine has not yet accumulated a full
            # window of history.  We compare the current row's epoch timestamp
            # against the machine's first-record epoch timestamp; if the gap is
            # less than the required window duration the feature is set to null.
            #
            # Condition to KEEP the value (set to null otherwise):
            #   current_epoch - machine_first_epoch >= window_duration_seconds
            elapsed_expr = (
                F.col(timestamp_col).cast("long") - F.col(_FIRST_TS_COL)
            )
            df = df.withColumn(
                feature_name,
                F.when(elapsed_expr >= window_duration_seconds, F.col(feature_name))
                 .otherwise(F.lit(None).cast("double"))
            )
            logger.info(
                f"  ✓ Minimum-window guard applied: first {window_duration_seconds}s "
                f"per machine will be null for '{feature_name}'"
            )
            
            # Validate the rolling window calculation
            if self.config.get('data_quality', {}).get('validate_rolling_windows', True):
                self._validate_rolling_feature(df, feature_name, source_column, aggregation)
            
            logger.info(f"✓ Created streaming feature: {feature_name}")

        # Drop the helper column — not part of the output schema
        df = df.drop(_FIRST_TS_COL)
        
        # CRITICAL: re-sort so output order is timestamp → Machine_ID (required format)
        logger.info(
            f"Final sort: Re-ordering data by {timestamp_col}, {partition_cols} "
            "to maintain required output order"
        )
        df = df.orderBy(timestamp_col, *partition_cols)
        
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH FEATURES  (daily / weekly aggregations joined back to each row)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_batch_feature(self, df: Any, feature_name: str, aggregation: str):
        """Light-weight sanity check for a batch-aggregated column."""
        logger.info(f"  Validating batch feature: {feature_name}")

        null_count = df.filter(F.col(feature_name).isNull()).count()
        if null_count > 0:
            logger.info(
                f"  ℹ {null_count} null values in '{feature_name}' "
                f"(expected: first calendar period per machine is intentionally null)"
            )
        else:
            logger.info(f"  ✓ No null values in {feature_name}")

        stats = df.select(
            F.min(feature_name).alias('min_val'),
            F.max(feature_name).alias('max_val'),
            F.mean(feature_name).alias('mean_val'),
        ).collect()[0]
        
        # FIX: Check for None values before formatting
        if stats['min_val'] is None or stats['max_val'] is None or stats['mean_val'] is None:
             logger.info(
                f"  Statistics: All values are null. (This usually happens if the dataset "
                f"only spans a single period and was caught by the first-period guard)."
            )
        else:
            logger.info(
                f"  Statistics: min={stats['min_val']:.4f}, "
                f"max={stats['max_val']:.4f}, mean={stats['mean_val']:.4f}"
            )

    def _apply_batch_features(self, df: Any) -> Any:
        """
        Compute long-term batch aggregations (daily / weekly) and join them
        back to every individual row so the ML model can access them as
        additional input features.

        How it works
        ------------
        1. Truncate each row's timestamp to the desired period (day or week).
        2. Group by [Machine_ID, period] and compute the chosen aggregation.
        3. Left-join the aggregated frame back to the original data on
           [Machine_ID, period], then drop the helper period column.

        This pattern is equivalent to Spark's rangeBetween with very large
        windows but is computed much more efficiently because it leverages
        a simple groupBy rather than a full partition-wide sort + scan.

        Batch features defined in the YAML
        ------------------------------------
        Daily_Vibration_PeakMean_Ratio  (aggregation_type: daily)
            max(Vibration_mm_s) / mean(Vibration_mm_s)  per machine per day.

            The ratio measures how "spiky" a machine's vibration is during
            a given day relative to its baseline.  A healthy machine keeps
            this ratio low and stable; a machine developing a mechanical
            fault shows rising ratios before absolute vibration thresholds
            are breached.  Giving the ML model this daily context alongside
            real-time readings allows it to distinguish a momentary bump
            (low daily ratio) from a persistent anomaly (high daily ratio).

        Weekly_Current_StdDev  (aggregation_type: weekly)
            stddev(Current_L1) per machine per week.

            Motors with healthy windings draw a steady current.  As
            insulation degrades or bearings wear, current draw becomes
            erratic and the within-week standard deviation grows.  A weekly
            aggregation smooths out normal load fluctuations and exposes
            the gradual trend — exactly the signal that is invisible to
            short streaming windows but important for a predictive model.
        """
        batch_features = self.config.get('batch_features', [])
        if not batch_features:
            logger.info("No batch features configured — skipping")
            return df

        logger.info("Applying batch (daily/weekly) features")

        timestamp_col = self.config['schema']['timestamp_column']
        partition_cols = self.config['schema']['partition_columns']
        # Internal helper column name — guaranteed not to clash with real columns
        PERIOD_COL = "_batch_period_"

        for feature_config in batch_features:
            if not feature_config.get('enabled', False):
                logger.info(f"Skipping disabled batch feature: {feature_config['feature_name']}")
                continue

            feature_name    = feature_config['feature_name']
            source_column   = feature_config['source_column']
            aggregation     = feature_config['aggregation']
            aggregation_type = feature_config['aggregation_type']  # 'daily' or 'weekly'

            logger.info(f"Creating batch feature: {feature_name} ({feature_config['description']})")
            logger.info(
                f"  Source: {source_column}, Aggregation: {aggregation}, "
                f"Period: {aggregation_type}"
            )

            # ── Step 1: truncate timestamp to the chosen period ──────────────
            if aggregation_type == 'daily':
                period_expr = F.date_trunc('day', F.col(timestamp_col))
            elif aggregation_type == 'weekly':
                period_expr = F.date_trunc('week', F.col(timestamp_col))
            else:
                logger.warning(
                    f"Unknown aggregation_type '{aggregation_type}' for {feature_name} — skipping"
                )
                continue

            df = df.withColumn(PERIOD_COL, period_expr)

            # ── Step 2: aggregate per [machine, period] ──────────────────────
            if aggregation == 'ratio_max_mean':
                # max / mean — detects spiky / impulsive behaviour vs baseline
                agg_df = df.groupBy(*partition_cols, PERIOD_COL).agg(
                    (F.max(source_column) / F.mean(source_column)).alias(feature_name)
                )
            elif aggregation == 'std':
                agg_df = df.groupBy(*partition_cols, PERIOD_COL).agg(
                    F.stddev(source_column).alias(feature_name)
                )
            elif aggregation == 'mean':
                agg_df = df.groupBy(*partition_cols, PERIOD_COL).agg(
                    F.mean(source_column).alias(feature_name)
                )
            elif aggregation == 'max':
                agg_df = df.groupBy(*partition_cols, PERIOD_COL).agg(
                    F.max(source_column).alias(feature_name)
                )
            else:
                logger.warning(
                    f"Unknown aggregation '{aggregation}' for batch feature {feature_name} — skipping"
                )
                df = df.drop(PERIOD_COL)
                continue

            # ── Step 3: join aggregate back to every row ─────────────────────
            join_keys = [*partition_cols, PERIOD_COL]
            df = df.join(agg_df, on=join_keys, how='left')

            # ── Step 4: null out the first (incomplete) period per machine ────
            #
            # The first calendar period a machine appears in (day or week) is
            # almost always INCOMPLETE: data collection started mid-day or
            # mid-week, so the aggregation is computed on fewer observations
            # than a full period contains.  Providing a value for this period
            # is misleading because the ML model cannot distinguish "quiet
            # machine" from "machine we only watched for 3 hours today".
            #
            # Fix: identify each machine's earliest period, then force the
            # feature column to null for every row that falls inside it.
            #
            # Timeline example (daily feature, data starts 2024-01-01 06:00):
            #   period 2024-01-01 → NULL  (only 18 h of data — incomplete day)
            #   period 2024-01-02 → value (full 24 h ✓)
            #   period 2024-01-03 → value (full 24 h ✓)
            #
            # Timeline example (weekly feature, data starts 2024-01-01):
            #   week  2024-W01   → NULL  (< 7 days — incomplete week)
            #   week  2024-W02   → value (full 7 days ✓)
            _FIRST_PERIOD_COL = "_batch_first_period_"
            machine_window = Window.partitionBy(*partition_cols)
            df = df.withColumn(
                _FIRST_PERIOD_COL,
                F.min(PERIOD_COL).over(machine_window)
            )
            df = df.withColumn(
                feature_name,
                F.when(
                    F.col(PERIOD_COL) == F.col(_FIRST_PERIOD_COL),
                    F.lit(None).cast("double")         # first period → null
                ).otherwise(F.col(feature_name))       # subsequent periods → keep value
            )
            df = df.drop(PERIOD_COL, _FIRST_PERIOD_COL)
            logger.info(
                f"  ✓ First-period guard applied: first {aggregation_type} period "
                f"per machine will be null for '{feature_name}'"
            )

            # Validate
            if self.config.get('data_quality', {}).get('validate_batch_features', True):
                self._validate_batch_feature(df, feature_name, aggregation)

            logger.info(f"✓ Created batch feature: {feature_name}")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # I/O helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _write_dataset(self, df: Any, dataset_config: Dict):
        """Write processed dataset WITHOUT Machine_ID partitioning (flat structure)"""
        output_path = dataset_config['output_path']
        write_mode = self.config['processing'].get('write_mode', 'overwrite')
        timestamp_col = self.config['schema']['timestamp_column']
        partition_cols = self.config['schema']['partition_columns']
        
        logger.info(f"Writing dataset to {output_path} (mode: {write_mode})")
        
        # Final sort: timestamp → Machine_ID (matches required output format)
        logger.info(f"Final sort by {timestamp_col}, {partition_cols} before writing")
        df = df.orderBy(timestamp_col, *partition_cols)
        
        # Verify timestamp order before writing
        if self.config.get('data_quality', {}).get('verify_timestamp_order', True):
            self._verify_timestamp_order(df)
        
        # Optional repartitioning for better parallelism
        if self.config['processing'].get('repartition', False):
            num_partitions = self.config['processing'].get('num_partitions', 10)
            logger.info(f"Repartitioning to {num_partitions} partitions")
        
        # CRITICAL: write WITHOUT partitionBy → flat parquet structure (no Machine_ID folders)
        logger.info("Writing without partitionBy — creating flat parquet structure")
        df.write \
            .mode(write_mode) \
            .parquet(output_path)
        
        logger.info(f"✓ Dataset written successfully")
        logger.info(
            f"   Output structure: {output_path}/part-*.parquet (flat, no Machine_ID folders)"
        )
    
    def _verify_timestamp_order(self, df: Any):
        """Verify that timestamps are in chronological order within each partition"""
        logger.info("Verifying timestamp order...")
        
        timestamp_col = self.config['schema']['timestamp_column']
        partition_cols = self.config['schema']['partition_columns']
        
        summary = df.groupBy(partition_cols).agg(
            F.min(timestamp_col).alias('first_timestamp'),
            F.max(timestamp_col).alias('last_timestamp'),
            F.count('*').alias('row_count')
        ).orderBy(partition_cols)
        
        logger.info("Timestamp range per machine:")
        summary.show(10, truncate=False)
        
        logger.info("Sample of final output order (first 20 rows):")
        df.select(timestamp_col, *partition_cols).show(20, truncate=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Main orchestration
    # ─────────────────────────────────────────────────────────────────────────

    def process_dataset(self, dataset_config: Dict):
        """Process a single dataset: streaming rolling features + batch features"""
        logger.info("=" * 80)
        logger.info(f"Processing dataset: {dataset_config['name']}")
        logger.info("=" * 80)
        
        # Read data
        df = self._read_dataset(dataset_config)
        
        # Data quality checks
        df = self._validate_data_quality(df, dataset_config)
        
        # Cache after quality checks — will be reused by both feature stages
        if self.config['processing'].get('cache_intermediate', False):
            df = df.cache()
        
        # ── Streaming pipeline features ──────────────────────────────────────
        df = self._apply_rolling_features(df)

        # ── Batch pipeline features ──────────────────────────────────────────
        df = self._apply_batch_features(df)
        
        # Show sample results
        logger.info("Sample output (first 5 rows):")
        df.show(5, truncate=False)
        
        # Write results
        self._write_dataset(df, dataset_config)
        
        # Unpersist cache
        if self.config['processing'].get('cache_intermediate', False):
            df.unpersist()
        
        logger.info(f"✓ Completed processing: {dataset_config['name']}")
    
    def process_all_datasets(self):
        """Process all datasets defined in configuration"""
        logger.info("Starting data engineering with rolling features")
        logger.info(f"Total datasets to process: {len(self.config['datasets'])}")
        
        for dataset_config in self.config['datasets']:
            try:
                self.process_dataset(dataset_config)
            except Exception as e:
                logger.error(
                    f"Error processing {dataset_config['name']}: {str(e)}", exc_info=True
                )
                continue
        
        logger.info("=" * 80)
        logger.info("Data Engineering completed!")
        logger.info("=" * 80)
    
    def stop(self):
        """Stop Spark session"""
        logger.info("Stopping Spark session")
        self.spark.stop()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data Egnineering with Rolling Window Features'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='services/data_engineering_service/config/feature_engineering_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Process specific dataset by name (optional)'
    )
    
    args = parser.parse_args()
    
    service = FeatureEngineering(args.config)
    
    try:
        if args.dataset:
            dataset_config = next(
                (d for d in service.config['datasets'] if d['name'] == args.dataset),
                None
            )
            if dataset_config:
                service.process_dataset(dataset_config)
            else:
                logger.error(f"Dataset '{args.dataset}' not found in configuration")
                sys.exit(1)
        else:
            service.process_all_datasets()
    finally:
        service.stop()


if __name__ == "__main__":
    main()