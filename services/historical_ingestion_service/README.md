# Historical Data Ingestion Service

A PySpark-based service for processing historical industrial washer data with configurable rolling window features for vibration monitoring and anomaly detection.

## Overview

This service ingests historical sensor data from industrial washers and computes rolling window aggregations (e.g., rolling max vibration over 10 minutes) to support predictive maintenance and anomaly detection use cases. The service is designed to handle large-scale time-series data with proper partitioning by machine and timestamp ordering.

## Features

- **Rolling Window Aggregations**: Compute time-based rolling features (max, mean, std, etc.) over configurable windows
- **Per-Machine Partitioning**: Rolling windows are calculated separately for each machine to avoid data contamination
- **Data Quality Validation**: Built-in checks for null timestamps, duplicates, and rolling window calculation accuracy
- **Flexible Configuration**: YAML-based configuration for datasets, features, and processing parameters
- **Scalable Processing**: Leverages Apache Spark for distributed processing of large datasets
- **Timestamp Ordering**: Maintains chronological order with proper sorting by timestamp and machine ID

## Architecture

```
Input Data (Parquet)
    ↓
Data Quality Checks (nulls, duplicates)
    ↓
Rolling Window Feature Engineering
    ↓
Validation (verify calculations)
    ↓
Output Data (Parquet, flat structure)
```


## Configuration

The service uses a YAML configuration file (`feature_engineering_config.yaml`) to define:

### 1. Datasets

```yaml
datasets:
  - name: "industrial_washer_normal"
    input_path: "data/synthetic_datasets/industrial_washer_normal"
    output_path: "data/processed_datasets/industrial_washer_normal_features"
    file_format: "parquet"
    has_labels: false
```

### 2. Schema Configuration

```yaml
schema:
  timestamp_column: "timestamp"
  partition_columns: ["Machine_ID"]  # Group rolling windows per machine
```

### 3. Rolling Features

```yaml
rolling_features:
  - feature_name: "Vibration_RollingMax_10min"
    description: "Maximum vibration in the last 10 minutes - Critical for shock detection"
    source_column: "Vibration_mm_s"
    aggregation: "max"
    window_duration: "10 minutes"
    window_type: "time_based"
    enabled: true
```

**Supported Window Durations:**
- Seconds: `"30 seconds"`, `"1 second"`
- Minutes: `"10 minutes"`, `"1 minute"`
- Hours: `"1 hour"`, `"2 hours"`
- Days: `"1 day"`, `"7 days"`

**Supported Aggregations:**
- `max`: Maximum value in the window
- Additional aggregations can be added (mean, std, min, etc.)

### 4. Processing Options

```yaml
processing:
  repartition: false
  num_partitions: 10
  cache_intermediate: true
  write_mode: "overwrite"  # Options: overwrite, append, error
```

### 5. Data Quality Checks

```yaml
data_quality:
  check_null_timestamps: true
  check_duplicate_timestamps: true
  validate_rolling_windows: true
  verify_timestamp_order: true
```

### 6. Spark Configuration

```yaml
spark_config:
  spark.sql.shuffle.partitions: 200
  spark.sql.adaptive.enabled: true
  spark.sql.adaptive.coalescePartitions.enabled: true
```

## Usage


### Programmatic Usage

```python
from hist_ingestion import HistoricalIngestionService

# Initialize service
service = HistoricalIngestionService('config/feature_engineering_config.yaml')

# Process all datasets
service.process_all_datasets()

# Or process a specific dataset
dataset_config = {
    'name': 'my_dataset',
    'input_path': 'data/input',
    'output_path': 'data/output',
    'file_format': 'parquet'
}
service.process_dataset(dataset_config)

# Clean up
service.stop()
```

## How Rolling Windows Work

The service implements **time-based rolling windows** using Spark's `Window.rangeBetween()` function:

```python
window_spec = (
    Window
    .partitionBy("Machine_ID")              # Separate windows per machine
    .orderBy(col("timestamp").cast("long")) # Order by Unix timestamp
    .rangeBetween(-window_seconds, 0)       # Look back N seconds
)
```

**Example**: For a 10-minute rolling max:
- At timestamp `2024-01-01 12:00:00`, the feature calculates the maximum vibration value from `11:50:00` to `12:00:00`
- Each machine has its own independent rolling window
- Windows slide continuously with each new timestamp

## Data Quality Validation

The service performs several validation checks:

### 1. Null Timestamp Check
Filters out rows with null timestamps to ensure valid time-series data.

### 2. Duplicate Timestamp Check
Removes duplicate (timestamp, Machine_ID) combinations to prevent data corruption.

### 3. Rolling Window Validation
For `max` aggregations, verifies that:
```
rolling_max(value) >= current_value
```

### 4. Timestamp Order Verification
Ensures output data is properly sorted by timestamp and Machine_ID.

## Output Structure

The service writes data in a **flat Parquet structure** without Machine_ID partitioning:

```
output_path/
├── part-00000-*.parquet
├── part-00001-*.parquet
├── part-00002-*.parquet
└── _SUCCESS
```

**Row Order**: Sorted by timestamp first, then Machine_ID
```
timestamp            Machine_ID  Vibration_mm_s  Vibration_RollingMax_10min
2024-01-01 00:00:00  1          2.5             2.5
2024-01-01 00:00:00  2          2.3             2.3
2024-01-01 00:00:00  3          2.7             2.7
```

## Logging

The service provides detailed logging at INFO level:

```
2024-01-15 10:30:00 - INFO - Loading configuration from config.yaml
2024-01-15 10:30:01 - INFO - Creating Spark session
2024-01-15 10:30:05 - INFO - Processing dataset: industrial_washer_normal
2024-01-15 10:30:06 - INFO - Read 1000000 rows from data/input
2024-01-15 10:30:07 - INFO - Applying rolling window features
2024-01-15 10:30:08 - INFO - Creating feature: Vibration_RollingMax_10min
2024-01-15 10:30:10 - INFO - ✓ Validation passed: All rolling max values >= source values
2024-01-15 10:30:12 - INFO - ✓ Dataset written successfully
```

## Performance Tuning

### For Large Datasets
```yaml
processing:
  repartition: true
  num_partitions: 200
  cache_intermediate: true

spark_config:
  spark.sql.shuffle.partitions: 400
  spark.driver.memory: 8g
  spark.executor.memory: 16g
```

### For Small Datasets
```yaml
processing:
  repartition: false
  cache_intermediate: false

spark_config:
  spark.sql.shuffle.partitions: 50
```

## Adding New Features

To add a new rolling feature, update the configuration:

```yaml
rolling_features:
  - feature_name: "Temperature_RollingAvg_30min"
    description: "Average temperature over 30 minutes"
    source_column: "Temperature_C"
    aggregation: "mean"  # Note: requires code update for mean aggregation
    window_duration: "30 minutes"
    enabled: true
```

To support new aggregations, modify the `_apply_rolling_features` method in `hist_ingestion.py`:

```python
if aggregation == 'max':
    df = df.withColumn(feature_name, F.max(source_column).over(window_spec))
elif aggregation == 'mean':
    df = df.withColumn(feature_name, F.avg(source_column).over(window_spec))
elif aggregation == 'std':
    df = df.withColumn(feature_name, F.stddev(source_column).over(window_spec))
```

## Error Handling

The service includes comprehensive error handling:

- **Invalid duration format**: Raises `ValueError` with clear message
- **Unsupported file format**: Raises `ValueError`
- **Missing dataset**: Logs error and exits with code 1
- **Processing errors**: Logs error with stack trace and continues to next dataset

## Use Cases

### 1. Predictive Maintenance
Monitor rolling max vibration to detect increasing wear patterns before failure.

### 2. Anomaly Detection
Compare current values against rolling statistics to identify outliers.

### 3. Feature Engineering for ML
Create rich time-series features for training machine learning models.

### 4. Historical Analysis
Aggregate historical sensor data for trend analysis and reporting.

## Best Practices

1. **Start with smaller windows** (5-10 minutes) and increase as needed
2. **Enable caching** for datasets you'll process multiple times
3. **Use appropriate partitioning** to balance parallelism and overhead
4. **Monitor Spark UI** for performance bottlenecks
5. **Validate output** by checking sample rows and statistics
6. **Keep features enabled/disabled** in config rather than deleting them

## Troubleshooting

### Issue: Out of Memory Errors
**Solution**: Reduce `num_partitions`, disable `cache_intermediate`, or increase executor memory

### Issue: Slow Processing
**Solution**: Enable `repartition`, increase `spark.sql.shuffle.partitions`, or reduce window sizes

### Issue: Incorrect Rolling Values
**Solution**: Check that `partition_columns` matches your machine identifier and verify timestamp format

### Issue: Unsorted Output
**Solution**: Ensure `verify_timestamp_order` is enabled and check final sort operations

---

**Generated for Machine Learning Engineers**  
Part of the Industrial Washing Machine Anomaly Detection System
