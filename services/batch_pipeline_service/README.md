# Batch Feature Pipeline for Washing Machine Anomaly Detection

## Overview

This batch pipeline computes **daily batch features** for the Feast feature store, specifically for washing machine anomaly detection. The pipeline is built with PySpark and follows the architecture established in your feature store configuration.

## Architecture

### Feature Separation Strategy

The system uses **two separate feature views** with different cadences:

```
┌─────────────────────────────────┐      ┌──────────────────────────────────┐
│  machine_streaming_features     │      │  machine_batch_features          │
│  (hourly/real-time updates)     │      │  (daily computation)             │
│                                 │      │                                  │
│  - Raw sensor readings          │      │  - Daily_Vibration_PeakMean_Ratio│
│  - Rolling-window features      │      │    (max/mean ratio per day)      │
│    (5min, 10min windows)        │      │                                  │
│  TTL: 12 hours                  │      │  TTL: 7 days                     │
└─────────────────────────────────┘      └──────────────────────────────────┘
```

**Why separate?**
- **Different refresh rates**: Streaming updates every few seconds; batch runs once per day
- **Independent TTLs**: Prevents evicting fresh streaming data while waiting for batch
- **Clear ownership**: Data engineers own batch features; streaming team owns real-time features
- **Scalability**: Batch pipelines can be optimized for historical data; streaming for latency

### Computed Feature

**`Daily_Vibration_PeakMean_Ratio`**

```
Formula: max(Vibration_mm_s) / mean(Vibration_mm_s)  [per machine, per day]

Interpretation:
  • High ratio (>1.5):   Spiky, impulsive vibration → potential mechanical fault
  • Normal ratio (1.0-1.5): Smooth operation → healthy machine
  • Low ratio (<1.0):    Shouldn't happen (max must be ≥ mean)
```

This feature captures **short-term shock events** relative to baseline vibration, which is a strong early indicator of bearing degradation or winding faults.

## File Structure

```
your_project/
├── batch_job.py                    # Main batch pipeline script
├── batch_config.yaml               # Configuration file (paths, Spark settings)
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- PySpark 3.3+
- PyYAML

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install pyspark==3.3.2
pip install pyyaml
```

## Configuration

The pipeline is configured via `batch_config.yaml`. Key sections:

### `paths`
```yaml
paths:
  data_warehouse_dir: "data/processed_datasets/industrial_washer_normal_features"
  offline_store_dir: "data/offline/machine_batch_features"
```
- **data_warehouse_dir**: Source of processed/enriched industrial washer data (output from your data engineering pipeline)
- **offline_store_dir**: Target directory for Feast offline store (must match `data_sources.py` path)

### `spark`
```yaml
spark:
  app_name: "batch-feature-pipeline-washing-machines"
  master: "local[*]"                    # "local[*]" for single machine
  partitions: 8                         # Number of output files
  configs:
    "spark.sql.session.timeZone": "UTC"
```

### `batch_features`
```yaml
batch_features:
  daily_vibration_ratio:
    enabled: true
    feature_name: "Daily_Vibration_PeakMean_Ratio"
    source_column: "Vibration_mm_s"
    aggregation_type: "daily"
```

## Usage

### Basic Run

```bash
python batch_job.py
```

The script expects `batch_config.yaml` in the current directory.

### Custom Config Path

If your config is elsewhere:

```bash
# Modify the config_path in main() function
config_path = "/path/to/your/batch_config.yaml"
```

Then run:
```bash
python batch_job.py
```

### Example Output

```
2024-01-15 10:30:45,123 - __main__ - INFO - ================================================================================
2024-01-15 10:30:45,124 - __main__ - INFO - BATCH FEATURE PIPELINE - WASHING MACHINE ANOMALY DETECTION
2024-01-15 10:30:45,125 - __main__ - INFO - ================================================================================
2024-01-15 10:30:45,200 - __main__ - INFO - Loading configuration from batch_config.yaml
2024-01-15 10:30:45,300 - __main__ - INFO - Initializing Spark session...
2024-01-15 10:30:50,500 - __main__ - INFO - Spark session initialized
2024-01-15 10:30:50,510 - __main__ - INFO - Reading input data from data warehouse...
2024-01-15 10:30:52,123 - __main__ - INFO - Loaded 450000 rows
2024-01-15 10:30:52,200 - __main__ - INFO - Computing daily batch features...
2024-01-15 10:30:58,456 - __main__ - INFO - Sample output (first 10 rows):
+-----------+-------------------+------------------+------------------------------+
|Machine_ID |timestamp          |Vibration_mm_s    |Daily_Vibration_PeakMean_Ratio|
+-----------+-------------------+------------------+------------------------------+
|1          |2024-01-15 06:15:30|3.456             |1.234                         |
|1          |2024-01-15 06:20:45|2.987             |1.234                         |
|2          |2024-01-15 07:05:12|5.123             |1.678                         |
...
2024-01-15 10:31:02,789 - __main__ - INFO - Writing offline features to: data/offline/machine_batch_features
2024-01-15 10:31:05,456 - __main__ - INFO - ✓ Data written successfully
2024-01-15 10:31:05,500 - __main__ - INFO - ================================================================================
2024-01-15 10:31:05,501 - __main__ - INFO - BATCH PIPELINE COMPLETED SUCCESSFULLY
2024-01-15 10:31:05,502 - __main__ - INFO - Output location: data/offline/machine_batch_features
2024-01-15 10:31:05,503 - __main__ - INFO - Features computed: Daily_Vibration_PeakMean_Ratio
2024-01-15 10:31:05,504 - __main__ - INFO - Suggested end-date for 'feast materialize-incremental' (UTC): 2024-01-15T10:31:05Z
```

## Integration with Feast

### Step 1: Run Batch Pipeline

```bash
python batch_job.py
```

This writes to `data/offline/machine_batch_features/part-*.parquet`

### Step 2: Materialize to Online Store

After the batch pipeline completes, materialize the offline features to your Redis online store:

```bash
# Using Feast CLI
feast materialize-incremental 2024-01-15T10:31:05Z

# Or from Python
from feast import FeatureStore
fs = FeatureStore()
fs.materialize_incremental(end_date=datetime(2024, 1, 15, 10, 31, 5))
```

### Step 3: Retrieve Features at Inference Time

```python
from feast import FeatureStore
from datetime import datetime

fs = FeatureStore()

# Request features for a specific machine
features_df = fs.get_online_features(
    features=[
        "machine_batch_features:Daily_Vibration_PeakMean_Ratio",
        "machine_streaming_features:Vibration_RollingMax_10min",
        "machine_streaming_features:Current_Imbalance_RollingMean_5min"
    ],
    entity_rows=[
        {"Machine_ID": 1},
        {"Machine_ID": 2},
        {"Machine_ID": 3}
    ]
)

# Returns all features (batch + streaming) in a single vector
print(features_df)
```

## Data Flow Diagram

```
┌──────────────────────────────────────────┐
│  Raw Industrial Washer Data              │
│  (sensors, timestamps, etc.)             │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  data_engineering.py (Your Data Pipeline)│
│  Computes streaming features             │
│  Rolling windows, derived columns, etc.  │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Processed Dataset (Data Warehouse)      │
│  data/processed_datasets/industrial_...  │
└────────────────────┬─────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   ┌─────────────┐        ┌──────────────────┐
   │  (Optional) │        │  batch_job.py    │
   │  Real-time  │        │  (This script)   │
   │  Streaming  │        │                  │
   │  Pipeline   │        │  Daily features  │
   └─────┬───────┘        └────────┬─────────┘
         │                         │
         │                    Writes to
         │                         │
         │    ┌────────────────────┘
         │    │
         ▼    ▼
    ┌──────────────────────┐
    │  Offline Store       │
    │  (Parquet files)     │
    │  data/offline/...    │
    └──────────┬───────────┘
               │
               │ feast materialize-incremental
               ▼
    ┌──────────────────────┐
    │  Online Store        │
    │  (Redis)             │
    │  redis:6379          │
    └──────────┬───────────┘
               │
               │ Feature retrieval at inference time
               ▼
    ┌──────────────────────┐
    │  ML Model            │
    │  Anomaly Detector    │
    │  (Predicts faults)   │
    └──────────────────────┘
```

## Troubleshooting

### Issue: FileNotFoundError - Input Parquet not found

**Cause**: The data warehouse directory path is incorrect or doesn't exist

**Solution**:
1. Verify the directory exists: `ls data/processed_datasets/industrial_washer_normal_features/`
2. Check the path in `batch_config.yaml`
3. Ensure the upstream data engineering pipeline has completed successfully

### Issue: Spark OutOfMemory error

**Cause**: Too many partitions or large dataset for available memory

**Solution**:
1. Reduce `spark_partitions` in `batch_config.yaml`
2. Reduce Spark executor memory in `configs`: adjust `spark.executor.memory`
3. Enable caching only if necessary: `cache_intermediate: true`

### Issue: Feature values are all NULL

**Cause**: First-period guard is too aggressive or data quality issues

**Solutions**:
1. Check that your data has complete daily coverage (not just partial days)
2. Verify timestamp column is correctly named and parsed
3. Review data quality logs for dropped rows due to null timestamps

### Issue: Parquet files not readable by Feast

**Cause**: Output format doesn't match Feast's expectations

**Solution**: 
- Verify partition structure: `ls -la data/offline/machine_batch_features/`
- Should show multiple `part-*.parquet` files (not subdirectories)
- Data types must match schema defined in `features.py`

## Performance Tuning

### For Large Datasets

```yaml
processing:
  repartition: true
  repartition_count: 32  # Increase for parallel writes

spark:
  partitions: 32
  configs:
    "spark.sql.shuffle.partitions": "64"
    "spark.driver.memory": "8g"
    "spark.executor.memory": "4g"
```

### For Local Development

```yaml
spark:
  partitions: 4
  master: "local[2]"
  configs:
    "spark.driver.memory": "2g"
```

## Related Files

- **`batch_config.yaml`**: Configuration file (modify paths and Spark settings here)
- **`features.py`**: Feast feature view definitions (must match output columns)
- **`data_sources.py`**: Feast data source configuration
- **`entity.py`**: Machine entity definition
- **`feature_services.py`**: Feature service grouping
- **`data_engineering.py`**: Your original data engineering pipeline (for reference)

## Next Steps

1. **Create** `batch_config.yaml` with your paths
2. **Run** the batch pipeline: `python batch_job.py`
3. **Verify** output in offline store directory
4. **Materialize** to online store with Feast CLI
5. **Retrieve** features in your inference code

## Support

For issues or questions:
1. Check logs in the console output
2. Verify configuration in `batch_config.yaml`
3. Review Spark UI at `http://localhost:4040` (during execution)
4. Check Feast documentation: https://docs.feast.dev/
