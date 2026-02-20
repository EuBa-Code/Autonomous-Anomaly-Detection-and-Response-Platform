"""
EXAMPLE USAGE - Industrial Washing Machine Dataset Generator

This script shows how to use the generator function and analyze the results.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, min, max, stddev
from services.create_datasets_service.src.industrial_washer_generator import generate_industrial_washer_datasets, save_datasets
from services.create_datasets_service.config import DATASETS_PATH

# ============================================================================
# 1. Initialize Spark
# ============================================================================

spark = SparkSession.builder \
    .appName("Washer Data Example") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ============================================================================
# 2. Generate Datasets
# ============================================================================

print("Generating datasets...")
normal_df, anomaly_df = generate_industrial_washer_datasets(
    spark=spark,
    num_rows=500_000,
    anomaly_rate=0.02
)

normal_streaming_df, anomaly_streaming_df = generate_industrial_washer_datasets(
    spark=spark,
    num_rows=10_000,
    anomaly_rate=0.02,
    streaming=True
)

# ============================================================================
# 3. Basic Data Exploration
# ============================================================================

print("\n" + "="*80)
print("DATASET SCHEMAS")
print("="*80)

print("\nNormal Dataset Schema:")
normal_df.printSchema()

print("\nAnomaly Dataset Schema:")
anomaly_df.printSchema()

print("\nNormal Dataset Streaming Schema:")
normal_streaming_df.printSchema()

print("\nAnomaly Dataset Streaming Schema:")
anomaly_streaming_df.printSchema()

# ============================================================================
# 4. Statistical Analysis
# ============================================================================

print("\n" + "="*80)
print("NORMAL DATASET - STATISTICAL SUMMARY")
print("="*80)
normal_df.describe().show()

print("\n" + "="*80)
print("ANOMALY DATASET - COMPARISON (Normal vs Anomaly)")
print("="*80)

# Compare normal vs anomalous records
print("\n--- CURRENT L1 Statistics ---")
anomaly_df.groupBy("is_anomaly").agg(
    count("*").alias("count"),
    avg("Current_L1").alias("avg_current_L1"),
    min("Current_L1").alias("min_current_L1"),
    max("Current_L1").alias("max_current_L1"),
    stddev("Current_L1").alias("stddev_current_L1")
).show()

print("\n--- VOLTAGE Statistics ---")
anomaly_df.groupBy("is_anomaly").agg(
    avg("Voltage_L_L").alias("avg_voltage"),
    min("Voltage_L_L").alias("min_voltage"),
    max("Voltage_L_L").alias("max_voltage"),
    stddev("Voltage_L_L").alias("stddev_voltage")
).show()

print("\n--- TEMPERATURE Statistics ---")
anomaly_df.groupBy("is_anomaly").agg(
    avg("Water_Temp_C").alias("avg_temp"),
    min("Water_Temp_C").alias("min_temp"),
    max("Water_Temp_C").alias("max_temp"),
    stddev("Water_Temp_C").alias("stddev_temp")
).show()

print("\n--- VIBRATION Statistics ---")
anomaly_df.groupBy("is_anomaly").agg(
    avg("Vibration_mm_s").alias("avg_vibration"),
    min("Vibration_mm_s").alias("min_vibration"),
    max("Vibration_mm_s").alias("max_vibration"),
    stddev("Vibration_mm_s").alias("stddev_vibration")
).show()

# ============================================================================
# 5. Cycle Phase Distribution
# ============================================================================

print("\n" + "="*80)
print("DISTRIBUTION BY CYCLE PHASE")
print("="*80)

phase_names = {
    0: "Idle",
    1: "Fill_Water",
    2: "Heating",
    3: "Wash",
    4: "Rinse",
    5: "Spin",
    6: "Drain"
}

print("\nNormal Dataset - Phase Distribution:")
normal_df.groupBy("Cycle_Phase_ID").count() \
    .orderBy("Cycle_Phase_ID").show()

print("\nAnomaly Dataset - Phase Distribution with Anomalies:")
anomaly_df.groupBy("Cycle_Phase_ID", "is_anomaly").count() \
    .orderBy("Cycle_Phase_ID", "is_anomaly").show()

print("\nNormal Dataset - Phase Distribution:")
normal_streaming_df.groupBy("Cycle_Phase_ID").count() \
    .orderBy("Cycle_Phase_ID").show()

print("\nAnomaly Dataset - Phase Distribution with Anomalies:")
anomaly_streaming_df.groupBy("Cycle_Phase_ID", "is_anomaly").count() \
    .orderBy("Cycle_Phase_ID", "is_anomaly").show()

# ============================================================================
# 6. Machine-Level Analysis
# ============================================================================

print("\n" + "="*80)
print("MACHINE-LEVEL ANALYSIS")
print("="*80)

print("\nMachines with Most Anomalies (Top 10):")
anomaly_df.filter(col("is_anomaly") == 1) \
    .groupBy("Machine_ID").count() \
    .orderBy(col("count").desc()) \
    .limit(10).show()

# ============================================================================
# 7. Sample Records Display
# ============================================================================

print("\n" + "="*80)
print("SAMPLE NORMAL RECORDS")
print("="*80)
normal_df.limit(10).show(truncate=False)

print("\n" + "="*80)
print("SAMPLE ANOMALOUS RECORDS (Overcurrent)")
print("="*80)
anomaly_df.filter(
    (col("is_anomaly") == 1) & (col("Current_L1") > 50)
).limit(5).show(truncate=False)

print("\n" + "="*80)
print("SAMPLE ANOMALOUS RECORDS (Voltage Issues)")
print("="*80)
anomaly_df.filter(
    (col("is_anomaly") == 1) & 
    ((col("Voltage_L_L") < 360) | (col("Voltage_L_L") > 440))
).limit(5).show(truncate=False)

print("\n" + "="*80)
print("SAMPLE ANOMALOUS RECORDS (Overheating)")
print("="*80)
anomaly_df.filter(
    (col("is_anomaly") == 1) & (col("Water_Temp_C") > 80)
).limit(5).show(truncate=False)

# ============================================================================
# 8. Save Datasets (Optional)
# ============================================================================

print("\n" + "="*80)
print("SAVING DATASETS")
print("="*80)

save_datasets(normal_df, anomaly_df)
save_datasets(normal_streaming_df, anomaly_streaming_df, streaming=True)

