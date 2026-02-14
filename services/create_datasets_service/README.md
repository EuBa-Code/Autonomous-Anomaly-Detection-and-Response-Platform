# Industrial Washing Machine Dataset Generator

PySpark-based synthetic data generator for **1,000,000 rows** of industrial washing machine sensor data with realistic anomalies.

## 📁 Project Structure

```
services/create_datasets_service/
├── src/
│   ├── __init__.py
│   ├── industrial_washer_generator.py   # Core generator logic
│   ├── example_usage.py                 # Complete usage examples
│   ├── spark_configs.py                 # Spark configurations for different environments
│   └── test_generator.py                # Quick validation tests (10K rows)
├── config/
│   ├── __init__.py
│   └── config.py                        
└── README.md                            
```

### File Descriptions

| File | Purpose |
|------|---------|
| `industrial_washer_generator.py` | Main generator with `generate_industrial_washer_datasets()` function |
| `example_usage.py` | Full demonstration: generate, analyze, save datasets |
| `test_generator.py` | Quick tests with 10K rows to validate generator |
| `spark_configs.py` | Pre-configured Spark sessions (local, cluster, AWS, etc.) |

## 🚀 Quick Start

### Generate Datasets

```bash
# Via Docker Compose (recommended)
docker-compose up create_datasets

# Or run directly
uv run -m services.create_datasets_service.src.example_usage
```

### Basic Usage

```python
from pyspark.sql import SparkSession
from industrial_washer_generator import generate_industrial_washer_datasets

# Initialize Spark
spark = SparkSession.builder \
    .appName("Washer Data") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Generate 1M rows with 2% anomalies
normal_df, anomaly_df = generate_industrial_washer_datasets(
    spark=spark,
    num_rows=1_000_000,
    anomaly_rate=0.02
)

normal_df.show(10)
```

### Test Before Full Generation

```bash
# Quick validation with 10K rows
python test_generator.py
```

## 📊 Datasets Generated

Creates **FOUR** datasets:

1. **Normal Historical** (1M rows) - Clean sensor data
2. **Anomaly Historical** (1M rows) - With 2% anomalies + `is_anomaly` label
3. **Normal Streaming** (100K rows) - Clean streaming data
4. **Anomaly Streaming** (100K rows) - With 2% anomalies + `is_anomaly` label

### Output Structure

```
data/synthetic_datasets/
├── industrial_washer_normal/                           # Historical Parquet (1M)
├── industrial_washer_with_anomalies/                   # Historical with labels (1M)
├── industrial_washer_normal_streaming/                 # Streaming Parquet (100K)
├── industrial_washer_with_anomalies_streaming/         # Streaming with labels (100K)
├── industrial_washer_normal_sample/                    # CSV sample (10K)
├── industrial_washer_with_anomalies_sample/            # CSV sample (10K)
├── industrial_washer_with_anomalies_sample_streaming   # CSV sample (10K)
└── industrial_washer_normal_sample_streaming/          # CSV sample (10K)
```

## 🔧 Features (12 Sensors + 1 Label)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `timestamp` | Recording timestamp | DateTime | 30-day span |
| `Machine_ID` | Machine identifier | Integer | 1-50 |
| `Cycle_Phase_ID` | Washing cycle phase | Integer | 0-6 |
| `Current_L1/L2/L3` | Three-phase current | Amperes | 2-35A |
| `Voltage_L_L` | Line-to-line voltage | Volts | 400V ±5% |
| `Water_Temp_C` | Water temperature | Celsius | 20-65°C |
| `Motor_RPM` | Motor speed | RPM | 0-1400 |
| `Water_Flow_L_min` | Water flow rate | L/min | 0-45 |
| `Vibration_mm_s` | Vibration level | mm/s | 0.5-8.5 |
| `Water_Pressure_Bar` | Water pressure | Bar | 0.1-2.8 |
| `is_anomaly` | Anomaly label | Integer | 0/1 |

## 🔄 Washing Cycle Phases

| ID | Phase | Current | RPM | Description |
|----|-------|---------|-----|-------------|
| 0 | Idle | ~2A | 0 | Standby |
| 1 | Fill_Water | ~8.5A | 0 | Water filling |
| 2 | Heating | ~35A | ~50 | Water heating |
| 3 | Wash | ~18.5A | ~80 | Active washing |
| 4 | Rinse | ~12A | ~70 | Rinsing |
| 5 | Spin | ~28A | ~1400 | High-speed spinning |
| 6 | Drain | ~6A | ~10 | Water draining |

## 🚨 Anomaly Types (2% of Data)

| Type | % | Description |
|------|---|-------------|
| **Overcurrent** | 30% | Current 2.5-4x higher (motor overload) |
| **Voltage Issues** | 25% | Drops (320-350V) or spikes (450-480V) |
| **Overheating** | 20% | Temperature 85-100°C (thermostat failure) |
| **Excess Vibration** | 15% | 15-25 mm/s (unbalanced load) |
| **Motor Malfunction** | 10% | Wrong RPM for phase |

## ⚙️ Configuration Options

### Custom Spark Configuration

```python
from spark_configs import get_spark_local_prod  # 4GB, optimized
# or get_spark_high_memory()  # 8GB for large datasets
# or get_spark_minimal()      # 1GB for constrained environments

spark = get_spark_local_prod()
normal_df, anomaly_df = generate_industrial_washer_datasets(spark, num_rows=1_000_000)
```

### Custom Parameters

```python
# Smaller dataset with more anomalies
normal_df, anomaly_df = generate_industrial_washer_datasets(
    spark=spark,
    num_rows=100_000,      # 100K rows
    anomaly_rate=0.05,     # 5% anomalies
    streaming=False        # Set True for streaming datasets
)
```

## 📈 Data Characteristics

- **Size**: 1M rows (historical), 100K rows (streaming)
- **Time Span**: 30 days (~2.6 seconds between readings)
- **Machines**: 50 industrial washers
- **Anomaly Rate**: 2% (configurable)
- **File Size**: ~200-300 MB (Parquet compressed)
- **Realistic**: Three-phase balance, voltage stability, phase-dependent parameters

## 🧪 Testing

```bash
# Run quick tests (10K rows)
uv run -m services/create_datasets_service/test_generator.py

## 🐳 Docker Usage

```bash
# Generate via docker-compose
docker-compose up create_datasets

# View logs
docker logs create_datasets

# Access Spark UI (while running)
http://localhost:4040
```

## 📚 Integration

```python
# Use in your ML pipeline
from industrial_washer_generator import generate_industrial_washer_datasets

spark = SparkSession.builder.appName("ML Pipeline").getOrCreate()
normal_df, anomaly_df = generate_industrial_washer_datasets(spark, num_rows=1_000_000)

# For training
X_train = anomaly_df.drop("is_anomaly")
y_train = anomaly_df.select("is_anomaly")

# For testing
X_test = normal_df  # Pure normal data for evaluation
```

## 🔍 Analysis Examples

See `example_usage.py` for comprehensive examples of:
- Statistical analysis by cycle phase
- Anomaly distribution analysis
- Machine-level anomaly tracking
- Comparison of normal vs anomalous readings

## 🛠️ Customization

Modify parameters in `industrial_washer_generator.py`:

```python
cycle_phases = {
    0: {"name": "Idle", "base_current": 2.0, "base_rpm": 0, ...},
    # Adjust parameters here
}
```

---

**Generated for Machine Learning Engineers**  
Part of the Industrial Washing Machine Anomaly Detection System
