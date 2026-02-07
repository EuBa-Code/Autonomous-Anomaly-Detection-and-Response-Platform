import logging
from typing import List, Optional
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, sqrt, abs as spark_abs, when, lit, variance, greatest, least, coalesce, count
)
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

class SparkDataPreprocessor:
    def __init__(
        self,
        enable_expensive_features: bool = True,
        pf_default: float = 0.95,
        sample_interval_seconds: float = 30.0,
        cache_threshold_rows: int = 1000,
        target_shuffle_partitions: str = "10",
    ):
        """
        Args:
            enable_expensive_features: If True, includes window-based features (slower)
            pf_default: default power factor to use if PF isn't measured
            sample_interval_seconds: reading interval (seconds), used for energy calculation
            cache_threshold_rows: threshold to trigger caching
            target_shuffle_partitions: temporary value for spark.sql.shuffle.partitions during window ops
        """
        self.enable_expensive_features = enable_expensive_features
        self.pf_default = float(pf_default)
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.cache_threshold_rows = int(cache_threshold_rows)
        self.target_shuffle_partitions = str(target_shuffle_partitions)
        self.derived_feature_cols: List[str] = []

    def _engineer_features(self, df: DataFrame) -> DataFrame:
        logger.info("Engineering derived features...")
        logger.info(f"Expensive features (windows): {'ENABLED' if self.enable_expensive_features else 'DISABLED'}")

        # 1. Current_Avg - average across three phases
        df = df.withColumn(
            "Current_Avg",
            (col("Current_L1") + col("Current_L2") + col("Current_L3")) / lit(3.0)
        )

        # 2. Apparent_Power = sqrt(3) * V_ll * I_avg
        df = df.withColumn(
            "Apparent_Power",
            sqrt(lit(3.0)) * col("Voltage_L_L") * col("Current_Avg")
        )

        # 3. Active_Power = sqrt(3) * V_ll * I_avg * PF
        df = df.withColumn(
            "Active_Power",
            sqrt(lit(3.0)) * col("Voltage_L_L") * col("Current_Avg") * lit(self.pf_default)
        )

        # 4. Reactive_Power computed from S and P: Q = sqrt(max(S^2 - P^2, 0))
        s_sq_minus_p_sq = (col("Apparent_Power") * col("Apparent_Power")) - (col("Active_Power") * col("Active_Power"))
        df = df.withColumn(
            "Reactive_Power",
            sqrt(when(s_sq_minus_p_sq >= 0.0, s_sq_minus_p_sq).otherwise(lit(0.0)))
        )

        # 5. Power_Factor = P / S (guarded)
        df = df.withColumn(
            "Power_Factor",
            when(col("Apparent_Power") > 0.0, col("Active_Power") / col("Apparent_Power")).otherwise(lit(0.0))
        )

        # 6. THD_Current (heuristic) -- keep coefficient as parameter if needed.
        df = df.withColumn(
            "THD_Current",
            col("THD_Voltage") * lit(1.2)
        )

        # 7. Current_P_to_P (Peak-to-Peak)
        df = df.withColumn(
            "Current_P_to_P",
            greatest(col("Current_L1"), col("Current_L2"), col("Current_L3")) -
            least(col("Current_L1"), col("Current_L2"), col("Current_L3"))
        )

        # 8. Max_Current_Instance
        df = df.withColumn(
            "Max_Current_Instance",
            greatest(col("Current_L1"), col("Current_L2"), col("Current_L3"))
        )

        # 9. Inrush_Peak (only during warmup)
        df = df.withColumn(
            "Inrush_Peak",
            when(col("Cycle_Phase_ID") == 1, col("Max_Current_Instance")).otherwise(lit(0.0))
        )

        # 10. Phase_Imbalance: mean absolute deviation relative to Current_Avg (in percent)
        mad = (
            (spark_abs(col("Current_L1") - col("Current_Avg")) +
             spark_abs(col("Current_L2") - col("Current_Avg")) +
             spark_abs(col("Current_L3") - col("Current_Avg"))) / lit(3.0)
        )
        # prevent division by zero using coalesce -> default denom = 1.0
        denom = coalesce(when(col("Current_Avg") > 0.0, col("Current_Avg")), lit(1.0))
        df = df.withColumn("Phase_Imbalance", (mad / denom) * lit(100.0))

        # 11. Energy per cycle [Wh] given sample interval
        seconds = lit(self.sample_interval_seconds)
        df = df.withColumn(
            "Energy_per_Cycle_Wh",
            col("Active_Power") * (seconds / lit(3600.0))
        )

        # Basic derived features (always included) — NOTE: matches actual column names
        self.derived_feature_cols = [
            "Current_Avg",
            "Apparent_Power",
            "Active_Power",
            "Reactive_Power",
            "Power_Factor",
            "THD_Current",
            "Current_P_to_P",
            "Max_Current_Instance",
            "Inrush_Peak",
            "Phase_Imbalance",
            "Energy_per_Cycle_Wh",
        ]

        # EXPENSIVE OPERATIONS (window-based)
        if self.enable_expensive_features:
            logger.info("Computing expensive window-based features...")

            spark_session = df.sparkSession
            original_partitions = spark_session.conf.get("spark.sql.shuffle.partitions")
            spark_session.conf.set("spark.sql.shuffle.partitions", self.target_shuffle_partitions)

            try:
                # Define window: last N readings. Example: 10 minutes with sample_interval_seconds
                readings_in_10min = int(round(600.0 / self.sample_interval_seconds))
                window_10min = Window.partitionBy("Machine_ID").orderBy(col("timestamp")).rowsBetween(-(readings_in_10min - 1), 0)

                df = df.withColumn(
                    "Power_Var_10min_raw",
                    variance(col("Active_Power")).over(window_10min)
                )

                df = df.withColumn(
                    "row_count_in_window",
                    count(col("Active_Power")).over(window_10min)
                )

                df = df.withColumn(
                        "Power_Var_10min",
                        when(col("row_count_in_window") >= readings_in_10min, col("Power_Var_10min_raw"))
                        .otherwise(lit(None))
                    ).drop("Power_Var_10min_raw", "row_count_in_window")

                self.derived_feature_cols.extend(["Power_Var_10min"])

            finally:
                spark_session.conf.set("spark.sql.shuffle.partitions", original_partitions)

        logger.info(f"Created {len(self.derived_feature_cols)} derived features")
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Engineers features and returns the DataFrame with all original columns + calculated features.
        NO SCALING OR ENCODING - just raw calculated values.
        """
        logger.info("Starting feature engineering (no scaling)...")

        # Decide about caching: if large, cache before heavy ops and materialize once.
        row_count = None
        try:
            row_count = df.count()
        except Exception as e:
            logger.warning(f"Could not count input DataFrame cheaply: {e}")

        if row_count is not None and row_count > self.cache_threshold_rows:
            logger.info(f"Input data: {row_count} rows -> caching for performance.")
            df = df.cache()
            # materialize cache once
            df.count()

        df_transformed = self._engineer_features(df)

        # Unpersist if we cached
        if row_count is not None and row_count > self.cache_threshold_rows:
            try:
                df.unpersist()
            except Exception:
                # fallback: call unpersist on transformed DF if needed
                try:
                    df_transformed.unpersist()
                except Exception:
                    pass

        return df_transformed
