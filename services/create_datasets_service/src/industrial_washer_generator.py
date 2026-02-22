"""
Industrial Washing Machine Dataset Generator
Generates x number rows of realistic sensor data
Creates TWO datasets: 
1. Normal dataset (no anomalies)
2. Dataset with 2% anomalies

MODIFICATIONS:
- 3 washing machines 
- All machines report data at the same timestamp (every second)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, rand, randn, expr, row_number, 
    unix_timestamp, from_unixtime, lit, round as spark_round, floor
)
from pyspark.sql.types import *
from pyspark.sql.window import Window

from services.create_datasets_service.config import DATASETS_PATH
def generate_industrial_washer_datasets(spark, num_rows=100_000, anomaly_rate=0.02, streaming : bool = False):
    """
    Generate two industrial washing machine sensor datasets.
    
    Parameters:
    -----------
    spark : SparkSession
        Active Spark session
    num_rows : int
        Number of rows to generate (default: 1,000,000)
    anomaly_rate : float
        Percentage of anomalies in the anomaly dataset (default: 0.02 = 2%)
    
    Returns:
    --------
    tuple: (normal_df, anomaly_df)
        - normal_df: DataFrame without anomalies
        - anomaly_df: DataFrame with 2% anomalies and is_anomaly label
    """
    
    print(f"Generating {num_rows:,} rows of industrial washing machine data...")
    print(f"Configuration: 3 machines, synchronized timestamps (1 second intervals)")
    
    # =========================================================================
    # STEP 1: Generate base normal dataset
    # =========================================================================
    
    # Create base dataframe with sequential IDs
    base_df = spark.range(num_rows).withColumn("row_id", col("id"))
    
    if streaming:
        # Path to normal dataset
        previous_data_path = f"{DATASETS_PATH}/industrial_washer_normal"
        
        try:
            # Read only timestamp column
            print(f"Reading last timestamp from: {previous_data_path}...")
            last_ts_row = spark.read.parquet(previous_data_path) \
                .select("timestamp") \
                .agg(max("timestamp").alias("max_ts")) \
                .collect()
            
            last_timestamp = last_ts_row[0]["max_ts"]
            print(f"Last timestamp found: {last_timestamp}")
            
            start_time_expr = unix_timestamp(lit(last_timestamp)) + 1
            
        except Exception as e:
            print(f"WARNING: Could not read previous dataset ({e}). Defaulting to current time.")
            from pyspark.sql.functions import current_timestamp
            start_time_expr = unix_timestamp(current_timestamp())
            
    else:
        # Training dataset
        start_time_expr = unix_timestamp(lit("2024-01-01 00:00:00"))

    # =========================================================================
    # STEP 1: Generate base normal dataset
    # =========================================================================
    
    # Create base dataframe with sequential IDs
    base_df = spark.range(num_rows).withColumn("row_id", col("id"))
    
    # Generate timestamp and Machine_ID
    df = base_df.select(
        # Timestamp advances every 10 rows (every second) relative to start_time_expr
        from_unixtime(
            start_time_expr + floor(col("row_id") / 3)
        ).alias("timestamp"),
        
        # 3 different industrial washing machines (cycles through 1-3 for each timestamp)
        ((col("row_id") % 3) + 1).alias("Machine_ID"),
        
        # Cycle phases: 0=Idle, 1=Fill, 2=Heat, 3=Wash, 4=Rinse, 5=Spin, 6=Drain
        (expr("floor(rand() * 7)")).cast("int").alias("Cycle_Phase_ID"),
        
        # Random seed for variations
        col("row_id")
    )
        
    
    # =========================================================================
    # STEP 2: Add realistic electrical and sensor parameters per cycle phase
    # =========================================================================
    
    # Current L1 (Amperes) - Base current depends on cycle phase
    df = df.withColumn(
        "Current_L1",
        spark_round(
            when(col("Cycle_Phase_ID") == 0, 2.0 + randn() * 0.3)      # Idle: ~2A
            .when(col("Cycle_Phase_ID") == 1, 8.5 + randn() * 0.4)     # Fill: ~8.5A
            .when(col("Cycle_Phase_ID") == 2, 35.0 + randn() * 1.5)    # Heat: ~35A (heating elements)
            .when(col("Cycle_Phase_ID") == 3, 18.5 + randn() * 0.8)    # Wash: ~18.5A
            .when(col("Cycle_Phase_ID") == 4, 12.0 + randn() * 0.6)    # Rinse: ~12A
            .when(col("Cycle_Phase_ID") == 5, 28.0 + randn() * 1.2)    # Spin: ~28A (high RPM motor)
            .otherwise(6.0 + randn() * 0.4),                            # Drain: ~6A
            2
        )
    )
    
    # Current L2 (Amperes) - Slight phase imbalance (±2-3% from L1)
    df = df.withColumn(
        "Current_L2",
        spark_round(col("Current_L1") * (1 + randn() * 0.025), 2)
    )
    
    # Current L3 (Amperes) - Slight phase imbalance (±2-3% from L1)
    df = df.withColumn(
        "Current_L3",
        spark_round(col("Current_L1") * (1 + randn() * 0.025), 2)
    )
    
    # Voltage Line-to-Line (Volts) - Industrial 3-phase: ~400V ±5%
    df = df.withColumn(
        "Voltage_L_L",
        spark_round(400.0 + randn() * 10.0, 1)
    )
    
    # Water Temperature (Celsius)
    df = df.withColumn(
        "Water_Temp_C",
        spark_round(
            when(col("Cycle_Phase_ID") == 0, 20.0 + randn() * 2)       # Idle: room temp
            .when(col("Cycle_Phase_ID") == 1, 22.0 + randn() * 2)      # Fill: cold water
            .when(col("Cycle_Phase_ID") == 2, 65.0 + randn() * 3)      # Heat: hot water
            .when(col("Cycle_Phase_ID") == 3, 63.0 + randn() * 2.5)    # Wash: hot
            .when(col("Cycle_Phase_ID") == 4, 30.0 + randn() * 3)      # Rinse: warm
            .when(col("Cycle_Phase_ID") == 5, 28.0 + randn() * 2)      # Spin: cooling down
            .otherwise(25.0 + randn() * 2),                             # Drain: cooling
            1
        )
    )
    
    # Motor RPM (Revolutions Per Minute)
    df = df.withColumn(
        "Motor_RPM",
        spark_round(
            when(col("Cycle_Phase_ID") == 0, 0.0)                       # Idle: motor off
            .when(col("Cycle_Phase_ID") == 1, 0.0)                      # Fill: motor off
            .when(col("Cycle_Phase_ID") == 2, 50.0 + randn() * 5)       # Heat: slow agitation
            .when(col("Cycle_Phase_ID") == 3, 80.0 + randn() * 8)       # Wash: medium speed
            .when(col("Cycle_Phase_ID") == 4, 70.0 + randn() * 7)       # Rinse: medium-low
            .when(col("Cycle_Phase_ID") == 5, 1400.0 + randn() * 50)    # Spin: very high speed
            .otherwise(10.0 + randn() * 3),                             # Drain: very slow
            0
        )
    )
    
    # Water Flow Rate (Liters/minute)
    df = df.withColumn(
        "Water_Flow_L_min",
        spark_round(
            when(col("Cycle_Phase_ID") == 0, 0.0)                       # Idle: no flow
            .when(col("Cycle_Phase_ID") == 1, 45.0 + randn() * 3)       # Fill: high flow
            .when(col("Cycle_Phase_ID") == 2, 5.0 + randn() * 1)        # Heat: minimal flow
            .when(col("Cycle_Phase_ID") == 3, 8.0 + randn() * 1.5)      # Wash: low flow
            .when(col("Cycle_Phase_ID") == 4, 40.0 + randn() * 3)       # Rinse: high flow
            .when(col("Cycle_Phase_ID") == 5, 0.0)                      # Spin: no flow
            .otherwise(0.0),                                             # Drain: no flow
            1
        )
    )
    
    # Vibration Level (mm/s) - Higher during spin cycle
    df = df.withColumn(
        "Vibration_mm_s",
        spark_round(
            when(col("Cycle_Phase_ID") == 0, 0.5 + rand() * 0.3)        # Idle: minimal
            .when(col("Cycle_Phase_ID") == 1, 0.8 + rand() * 0.4)       # Fill: low
            .when(col("Cycle_Phase_ID") == 2, 1.2 + rand() * 0.5)       # Heat: low-medium
            .when(col("Cycle_Phase_ID") == 3, 2.5 + rand() * 0.8)       # Wash: medium
            .when(col("Cycle_Phase_ID") == 4, 2.0 + rand() * 0.7)       # Rinse: medium-low
            .when(col("Cycle_Phase_ID") == 5, 8.5 + rand() * 1.5)       # Spin: HIGH vibration
            .otherwise(1.0 + rand() * 0.4),                              # Drain: low
            2
        )
    )
    
    # Water Pressure (Bar)
    df = df.withColumn(
        "Water_Pressure_Bar",
        spark_round(
            when(col("Cycle_Phase_ID") == 0, 0.1 + rand() * 0.05)       # Idle: minimal
            .when(col("Cycle_Phase_ID") == 1, 2.8 + randn() * 0.15)     # Fill: normal pressure
            .when(col("Cycle_Phase_ID") == 2, 1.2 + randn() * 0.1)      # Heat: lower
            .when(col("Cycle_Phase_ID") == 3, 1.5 + randn() * 0.12)     # Wash: medium
            .when(col("Cycle_Phase_ID") == 4, 2.5 + randn() * 0.15)     # Rinse: high
            .when(col("Cycle_Phase_ID") == 5, 0.2 + rand() * 0.1)       # Spin: very low
            .otherwise(0.1 + rand() * 0.05),                             # Drain: minimal
            2
        )
    )
    
    # Drop temporary columns
    normal_df = df.drop("row_id", "id")
    
    print(f"✓ Normal dataset created: {normal_df.count():,} rows")
    
    # =========================================================================
    # STEP 3: Create anomaly dataset based on normal dataset
    # =========================================================================
    
    # Add anomaly marker
    anomaly_df = normal_df.withColumn(
        "anomaly_random", 
        rand()
    ).withColumn(
        "is_anomaly",
        (col("anomaly_random") < anomaly_rate).cast("int")
    )
    
    # =========================================================================
    # STEP 4: Inject different types of anomalies (2% of data)
    # =========================================================================
    
    # Anomaly Type 1: Extreme overcurrent (30% of anomalies)
    anomaly_df = anomaly_df.withColumn(
        "Current_L1",
        when(
            (col("is_anomaly") == 1) & (col("anomaly_random") < anomaly_rate * 0.3),
            spark_round(col("Current_L1") * (2.5 + rand() * 1.5), 2)  # 2.5x to 4x overcurrent
        ).otherwise(col("Current_L1"))
    ).withColumn(
        "Current_L2",
        when(
            (col("is_anomaly") == 1) & (col("anomaly_random") < anomaly_rate * 0.3),
            spark_round(col("Current_L2") * (2.5 + rand() * 1.5), 2)
        ).otherwise(col("Current_L2"))
    ).withColumn(
        "Current_L3",
        when(
            (col("is_anomaly") == 1) & (col("anomaly_random") < anomaly_rate * 0.3),
            spark_round(col("Current_L3") * (2.5 + rand() * 1.5), 2)
        ).otherwise(col("Current_L3"))
    )
    
    # Anomaly Type 2: Voltage drops/spikes (25% of anomalies)
    anomaly_df = anomaly_df.withColumn(
        "Voltage_L_L",
        when(
            (col("is_anomaly") == 1) & 
            (col("anomaly_random") >= anomaly_rate * 0.3) & 
            (col("anomaly_random") < anomaly_rate * 0.55),
            # Severe voltage drop (320-350V) or spike (450-480V)
            when(rand() > 0.5, 
                 spark_round(320.0 + rand() * 30, 1)  # Drop
            ).otherwise(
                 spark_round(450.0 + rand() * 30, 1)  # Spike
            )
        ).otherwise(col("Voltage_L_L"))
    )
    
    # Anomaly Type 3: Overheating (20% of anomalies)
    anomaly_df = anomaly_df.withColumn(
        "Water_Temp_C",
        when(
            (col("is_anomaly") == 1) & 
            (col("anomaly_random") >= anomaly_rate * 0.55) & 
            (col("anomaly_random") < anomaly_rate * 0.75),
            spark_round(85.0 + rand() * 15, 1)  # Dangerous temperature (85-100°C)
        ).otherwise(col("Water_Temp_C"))
    )
    
    # Anomaly Type 4: Excessive vibration (15% of anomalies)
    anomaly_df = anomaly_df.withColumn(
        "Vibration_mm_s",
        when(
            (col("is_anomaly") == 1) & 
            (col("anomaly_random") >= anomaly_rate * 0.75) & 
            (col("anomaly_random") < anomaly_rate * 0.90),
            spark_round(15.0 + rand() * 10, 2)  # Extreme vibration (15-25 mm/s)
        ).otherwise(col("Vibration_mm_s"))
    )
    
    # Anomaly Type 5: Motor malfunction - wrong RPM for phase (10% of anomalies)
    anomaly_df = anomaly_df.withColumn(
        "Motor_RPM",
        when(
            (col("is_anomaly") == 1) & 
            (col("anomaly_random") >= anomaly_rate * 0.90),
            # Motor stuck at wrong speed or spinning when it shouldn't
            when(col("Cycle_Phase_ID").isin([0, 1]), 
                 spark_round(800.0 + rand() * 400, 0)  # Motor running when should be off
            ).otherwise(
                 spark_round(50.0 + rand() * 30, 0)    # Motor too slow
            )
        ).otherwise(col("Motor_RPM"))
    )
    
    # Drop temporary columns
    anomaly_df = anomaly_df.drop("anomaly_random")
    
    # Reorder columns
    column_order = [
        "timestamp", "Machine_ID", "Cycle_Phase_ID",
        "Current_L1", "Current_L2", "Current_L3", "Voltage_L_L",
        "Water_Temp_C", "Motor_RPM", "Water_Flow_L_min",
        "Vibration_mm_s", "Water_Pressure_Bar", "is_anomaly"
    ]
    
    anomaly_df = anomaly_df.select(*column_order)
    
    # Calculate anomaly statistics
    anomaly_count = anomaly_df.filter(col("is_anomaly") == 1).count()
    anomaly_percentage = (anomaly_count / num_rows) * 100
    
    print(f"✓ Anomaly dataset created: {anomaly_df.count():,} rows")
    print(f"  - Normal records: {num_rows - anomaly_count:,}")
    print(f"  - Anomalous records: {anomaly_count:,} ({anomaly_percentage:.2f}%)")
    print(f"  - Unique timestamps: {num_rows // 3:,} (3 machines per timestamp)")
    
    return normal_df, anomaly_df


def save_datasets(normal_df, anomaly_df, output_path=DATASETS_PATH, streaming : bool = False):
    """
    Save both datasets to Parquet format.
    
    Parameters:
    -----------
    normal_df : DataFrame
        Normal dataset without anomalies
    anomaly_df : DataFrame
        Dataset with anomalies
    output_path : str
        Base path for output files
    """
    
    print(f"\nSaving datasets to {output_path}...")
    
    if not streaming:
    # Save normal dataset
        normal_path = f"{output_path}/industrial_washer_normal"
        normal_df.write.mode("overwrite").parquet(normal_path)
        print(f"✓ Normal dataset saved to: {normal_path}")
        
        # Save anomaly dataset
        anomaly_path = f"{output_path}/industrial_washer_with_anomalies"
        anomaly_df.write.mode("overwrite").parquet(anomaly_path)
        print(f"✓ Anomaly dataset saved to: {anomaly_path}")
        
        # Also save as CSV for easier inspection (only first 10K rows to save space)
        print("\nSaving sample CSV files (first 10,000 rows)...")
        
        normal_df.limit(10000).coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/industrial_washer_normal_sample")
        print(f"✓ Normal sample CSV saved")
        
        anomaly_df.limit(10000).coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/industrial_washer_with_anomalies_sample")
        print(f"✓ Anomaly sample CSV saved")

    else:
        normal_path = f"{output_path}/industrial_washer_normal_streaming"
        normal_df.write.mode("overwrite").parquet(normal_path)
        print(f"✓ Normal dataset streaming saved to: {normal_path}")
        
        # Save anomaly dataset
        anomaly_path = f"{output_path}/industrial_washer_with_anomalies_streaming"
        anomaly_df.write.mode("overwrite").parquet(anomaly_path)
        print(f"✓ Anomaly dataset streaming saved to: {anomaly_path}")
        
        # Also save as CSV for easier inspection (only first 10K rows to save space)
        print("\nSaving sample CSV files (first 10,000 rows)...")
        
        normal_df.limit(10000).coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/industrial_washer_normal_sample_streaming")
        print(f"✓ Normal sample streaming CSV saved")
        
        anomaly_df.limit(10000).coalesce(1).write.mode("overwrite") \
            .option("header", "true") \
            .csv(f"{output_path}/industrial_washer_with_anomalies_sample_streaming")
        print(f"✓ Anomaly sample streaming CSV saved")

def display_sample_data(normal_df, anomaly_df):
    """Display sample records from both datasets."""
    
    print("\n" + "="*80)
    print("NORMAL DATASET - Sample Records")
    print("="*80)
    normal_df.show(10, truncate=False)
    
    print("\n" + "="*80)
    print("ANOMALY DATASET - Sample Normal Records")
    print("="*80)
    anomaly_df.filter(col("is_anomaly") == 0).show(5, truncate=False)
    
    print("\n" + "="*80)
    print("ANOMALY DATASET - Sample Anomalous Records")
    print("="*80)
    anomaly_df.filter(col("is_anomaly") == 1).show(10, truncate=False)
    
    print("\n" + "="*80)
    print("STATISTICS BY CYCLE PHASE")
    print("="*80)
    anomaly_df.groupBy("Cycle_Phase_ID", "is_anomaly").count() \
        .orderBy("Cycle_Phase_ID", "is_anomaly").show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Industrial Washer Data Generator") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    print("\n" + "="*80)
    print("INDUSTRIAL WASHING MACHINE DATASET GENERATOR")
    print("="*80 + "\n")
    
    # Generate datasets
    normal_df, anomaly_df = generate_industrial_washer_datasets(
        spark=spark,
        num_rows=1_000_000,
        anomaly_rate=0.02
    )
    
    # Display sample data
    display_sample_data(normal_df, anomaly_df)
    
    # Save datasets
    save_datasets(normal_df, anomaly_df)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print("\nDatasets ready for anomaly detection model training!")
    print("- Use 'normal_df' for baseline/unsupervised learning")
    print("- Use 'anomaly_df' for supervised anomaly detection")
    
    # Keep Spark session alive for interactive use
    # spark.stop()  # Uncomment to stop Spark session