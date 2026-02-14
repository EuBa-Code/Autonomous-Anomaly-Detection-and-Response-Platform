# Industrial Washing Machine Dataset Generator

PySpark-based synthetic data generator for **1,000,000 rows** of industrial washing machine sensor data with realistic anomalies for anomaly detection model training.

## 📋 Overview

This generator creates **FOUR datasets**:

1. **Normal Historical Dataset**: Clean sensor data without anomalies
2. **Anomaly Historical Dataset**: Same data with 2% injected anomalies + `is_anomaly` label (supervised learning)
3. **Normal Streaming Dataset**: Clean sensor data without anomalies
4. **Anomaly Streaming Dataset**: Same data with 2% injected anomalies + `is_anomaly` label (supervised learning)

## 🔧 Features Generated

| Feature | Description | Unit | Normal Range |
|---------|-------------|------|--------------|
| `timestamp` | Recording timestamp | DateTime | 30-day span |
| `Machine_ID` | Machine identifier | Integer | 1-50 |
| `Cycle_Phase_ID` | Washing cycle phase | Integer | 0-6 |
| `Current_L1` | Line 1 current | Amperes | 2-35A (phase-dependent) |
| `Current_L2` | Line 2 current | Amperes | 2-35A (±2-3% from L1) |
| `Current_L3` | Line 3 current | Amperes | 2-35A (±2-3% from L1) |
| `Voltage_L_L` | Line-to-line voltage | Volts | 400V ±5% |
| `Water_Temp_C` | Water temperature | Celsius | 20-65°C (phase-dependent) |
| `Motor_RPM` | Motor speed | RPM | 0-1400 (phase-dependent) |
| `Water_Flow_L_min` | Water flow rate | L/min | 0-45 (phase-dependent) |
| `Vibration_mm_s` | Vibration level | mm/s | 0.5-8.5 (phase-dependent) |
| `Water_Pressure_Bar` | Water pressure | Bar | 0.1-2.8 (phase-dependent) |
| `is_anomaly` | Anomaly label | Integer | 0 (normal) / 1 (anomaly) |

## 🔄 Washing Cycle Phases

| Phase ID | Phase Name | Description | Typical Current | Typical RPM |
|----------|-----------|-------------|-----------------|-------------|
| 0 | Idle | Machine standby | ~2A | 0 |
| 1 | Fill_Water | Water filling | ~8.5A | 0 |
| 2 | Heating | Water heating | ~35A | ~50 |
| 3 | Wash | Active washing | ~18.5A | ~80 |
| 4 | Rinse | Rinsing cycle | ~12A | ~70 |
| 5 | Spin | High-speed spinning | ~28A | ~1400 |
| 6 | Drain | Water draining | ~6A | ~10 |

## 🚨 Anomaly Types (2% of data)

The generator injects 5 types of realistic anomalies:

1. **Overcurrent** (30% of anomalies)
   - Current 2.5-4x higher than normal
   - Indicates motor overload or short circuit

2. **Voltage Issues** (25% of anomalies)
   - Severe drops: 320-350V
   - Spikes: 450-480V
   - Power supply problems

3. **Overheating** (20% of anomalies)
   - Temperature: 85-100°C
   - Thermostat failure or heating element malfunction

4. **Excessive Vibration** (15% of anomalies)
   - Vibration: 15-25 mm/s
   - Unbalanced load or bearing failure

5. **Motor Malfunction** (10% of anomalies)
   - Wrong RPM for cycle phase
   - Motor running when should be off
   - Motor speed too low

## 🚀 Quick Start

### Installation

```python
# Requires PySpark
make create datasets
```

### Basic Usage (See example_usage.py)

```python
from pyspark.sql import SparkSession
from industrial_washer_generator import generate_industrial_washer_datasets

# Initialize Spark
spark = SparkSession.builder \
    .appName("Washer Data") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Generate datasets
normal_df, anomaly_df = generate_industrial_washer_datasets(
    spark=spark,
    num_rows=1_000_000,
    anomaly_rate=0.02,  # 2% anomalies
    streaming=False # True for streaming datasets
)

# Display samples
normal_df.show(10)
anomaly_df.filter("is_anomaly = 1").show(10)
```

### Save Datasets

```python
from industrial_washer_generator import save_datasets

# Saves to Parquet + CSV samples
save_datasets(normal_df, anomaly_df, output_path="/your/path", streaming=False)
```

## 📊 Example Analysis

### Check Anomaly Distribution

```python
anomaly_df.groupBy("is_anomaly").count().show()

# +----------+------+
# |is_anomaly| count|
# +----------+------+
# |         0|980000|
# |         1| 20000|
# +----------+------+
```

### Compare Normal vs Anomalous Statistics

```python
anomaly_df.groupBy("is_anomaly").agg(
    avg("Current_L1").alias("avg_current"),
    max("Current_L1").alias("max_current"),
    avg("Voltage_L_L").alias("avg_voltage"),
    avg("Water_Temp_C").alias("avg_temp")
).show()
```

### Find Specific Anomaly Types

```python
# Overcurrent anomalies
anomaly_df.filter("is_anomaly = 1 AND Current_L1 > 50").show()

# Overvoltage anomalies
anomaly_df.filter("is_anomaly = 1 AND Voltage_L_L > 440").show()

# Overheating anomalies
anomaly_df.filter("is_anomaly = 1 AND Water_Temp_C > 80").show()
```

### Train/Test Split

```python
# 80/20 split
train_df, test_df = anomaly_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training: {train_df.count():,} rows")
print(f"Test: {test_df.count():,} rows")
```

## 🎯 Use Cases

### 1. Supervised Anomaly Detection

```python
# Use anomaly_df with is_anomaly label
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Prepare features
feature_cols = ["Current_L1", "Current_L2", "Current_L3", 
                "Voltage_L_L", "Water_Temp_C", "Motor_RPM",
                "Vibration_mm_s"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(anomaly_df)

# Train model
rf = RandomForestClassifier(labelCol="is_anomaly", featuresCol="features")
model = rf.fit(data)
```

### 2. Unsupervised Anomaly Detection

```python
# Use normal_df for baseline
# Train on normal data, detect anomalies as outliers
from pyspark.ml.clustering import KMeans

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
normal_data = assembler.transform(normal_df)

kmeans = KMeans(k=7, seed=42)  # 7 clusters (one per phase)
model = kmeans.fit(normal_data)
```

### 3. Time Series Analysis

```python
# Analyze patterns over time
from pyspark.sql.functions import hour, dayofweek

anomaly_df.withColumn("hour", hour("timestamp")) \
    .withColumn("day", dayofweek("timestamp")) \
    .groupBy("hour", "is_anomaly").count() \
    .orderBy("hour").show()
```

### 4. Predictive Maintenance

```python
# Identify machines with frequent anomalies
from pyspark.sql.functions import count, sum as spark_sum

machine_anomalies = anomaly_df.groupBy("Machine_ID").agg(
    count("*").alias("total_records"),
    spark_sum("is_anomaly").alias("anomaly_count")
).withColumn(
    "anomaly_rate", 
    col("anomaly_count") / col("total_records") * 100
).orderBy(col("anomaly_rate").desc())

machine_anomalies.show(10)
```

## 📁 Output Structure

When saved, creates:

```
datasets/
├── industrial_washer_normal/                    # Parquet (1M rows)
│   ├── part-00000-*.parquet
│   └── _SUCCESS
├── industrial_washer_with_anomalies/           # Parquet (1M rows + labels)
│   ├── part-00000-*.parquet
│   └── _SUCCESS
├── industrial_washer_normal_sample/            # CSV sample (10K rows)
│   └── part-00000-*.csv
└── industrial_washer_with_anomalies_sample/    # CSV sample (10K rows)
    └── part-00000-*.csv
```

## 🔬 Data Characteristics

- **Size**: 1,000,000 rows
- **Time Span**: 30 days (~2.6 seconds between readings)
- **Machines**: 50 industrial washers
- **Anomaly Rate**: 2% (20,000 anomalous records)
- **Features**: 13 columns (12 sensors + 1 label)
- **File Size**: ~200-300 MB (Parquet compressed)

## 📝 Notes

### Realistic Physical Relationships

- Three-phase current balance (L1, L2, L3 within ±3%)
- Voltage stability around 400V industrial standard
- Temperature correlates with heating phase
- RPM increases during spin cycle
- Vibration peaks during high-speed spinning
- Water flow active during fill/rinse phases

### Anomaly Design Principles

- Anomalies reflect real industrial failures
- Multiple sensor correlations broken during anomalies
- Severity varies (some subtle, some extreme)
- Anomalies distributed across all cycle phases
- Machine-independent anomaly distribution

## 🛠️ Customization

Adjust generation parameters:

```python
# Generate smaller dataset with more anomalies
normal_df, anomaly_df = generate_industrial_washer_datasets(
    spark=spark,
    num_rows=100_000,      # 100K rows
    anomaly_rate=0.05      # 5% anomalies
)
```

Modify cycle phase parameters in `industrial_washer_generator.py`:

```python
cycle_phases = {
    0: {"name": "Idle", "base_current": 2.0, ...},
    # Customize parameters here
}
```

## 📚 References

- Industrial 3-phase power: 400V line-to-line
- Typical industrial washer specs
- Common failure modes in industrial equipment
- Sensor data characteristics

## 🤝 Contributing

Feel free to:
- Add new anomaly types
- Include additional sensors
- Modify physical parameters
- Enhance documentation

## 📄 License

Free to use for research, education, and commercial ML projects.

## 🎓 Example Applications

1. **Anomaly Detection Models**: Train classifiers (RF, XGBoost, Neural Networks)
2. **Predictive Maintenance**: Identify failing machines before breakdown
3. **Time Series Forecasting**: Predict sensor values
4. **Feature Engineering**: Create derived features
5. **Explainable AI**: Analyze which features indicate anomalies
6. **Deep Learning**: Train autoencoders for unsupervised detection

---

**Generated with ❤️ for Machine Learning Engineers and Data Scientists**

For questions or issues, refer to `example_usage.py` for comprehensive examples.
