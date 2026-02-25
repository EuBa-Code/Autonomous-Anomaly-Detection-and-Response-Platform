# Data Engineering Service

A PySpark-based feature engineering service for industrial washer sensor data. It processes raw time-series readings through a dual-pipeline architecture — a **streaming pipeline** for short-term rolling window features and a **batch pipeline** for daily/weekly aggregations — producing enriched datasets for downstream anomaly detection ML models.

---

## Project Structure

```
data_engineering_service/
├── config/
│   └── feature_engineering_config.yaml   # All pipeline configuration (datasets, features, Spark tuning)
└── src/
    └── data_engineering.py               # HistoricalIngestionService class + main entry point
```

---

## Quick Start

```bash
# Run data ing
make data_engineering
```

**Default config path** (used when `--config` is omitted):
```
services/data_engineering_service/config/feature_engineering_config.yaml
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyspark` | Distributed data processing and window functions |
| `pyyaml` | YAML config loading |

---

## Architecture Overview

The service is orchestrated by the `HistoricalIngestionService` class. For each dataset, it runs the following steps in order:

```
Read Parquet
    └─▶ Data Quality Checks
            └─▶ Cache DataFrame
                    ├─▶ [Streaming Pipeline] Derived Columns → Rolling Window Features
                    └─▶ [Batch Pipeline]     Period Truncation → GroupBy Agg → Left Join
                                └─▶ Final Sort (timestamp → Machine_ID)
                                        └─▶ Write Parquet (flat, no partitioning)
```

### Streaming Pipeline (short-term rolling windows)

Rolling features are computed using Spark's `Window.rangeBetween` partitioned by `Machine_ID`. Each window looks backwards in time from the current row, keeping each machine's computation independent. Before the window is applied, `_compute_derived_columns()` pre-calculates any composite signals (e.g. three-phase current imbalance) that cannot be expressed as a single raw column reference.

Window duration strings (e.g. `"10 minutes"`) are parsed into seconds internally and passed to `rangeBetween`. Supported units: `seconds`, `minutes`, `hours`, `days`.

### Batch Pipeline (daily / weekly aggregations)

Batch features use a more efficient pattern than large-range window functions. For each feature:

1. The timestamp is **truncated** to the configured period (`day` or `week`) using `date_trunc`.
2. A `groupBy([Machine_ID, period])` computes the aggregation (e.g. `max/mean`, `stddev`).
3. The result is **left-joined** back to every individual row on `[Machine_ID, period]`, so the ML model receives both the raw sensor value and the broader daily/weekly context in the same record.

This approach is cheaper than a full partition-wide sort + scan and produces the same output.

---

## Output Columns

The service appends the following columns to every row in the output dataset. The first is an intermediate derived scalar used as input to a rolling feature; the remaining four are the final engineered features written to Parquet.

---

### `Current_Imbalance_Ratio` *(intermediate derived column)*

**Formula:**
```
(max(Current_L1, Current_L2, Current_L3) - min(Current_L1, Current_L2, Current_L3))
─────────────────────────────────────────────────────────────────────────────────────
               mean(Current_L1, Current_L2, Current_L3)
```

**Type:** `float` — dimensionless ratio  
**Computed by:** `_compute_derived_columns()`  
**Appears in output:** Yes (retained alongside the rolling feature derived from it)

This column captures how unequal the three-phase motor current draw is at any given instant. In a perfectly balanced three-phase motor the three currents are identical and the ratio is 0. As the motor develops faults — winding insulation breakdown, bearing wear, partial phase loss — the phases start drawing unequal current and the ratio rises. A value above ~0.02 is a recognised early-warning threshold in motor health monitoring.

The ratio is computed as a plain scalar column first, rather than being embedded directly into the rolling window expression, so that it can be inspected, validated, and reused by multiple rolling features without recalculating the formula each time.

**Why use it:** Electrical imbalance precedes mechanical symptoms. This derived signal gives the ML model an early warning channel that acts before vibration or temperature readings change significantly.

---

### `Vibration_RollingMax_10min` *(streaming feature)*

**Formula:**
```
max(Vibration_mm_s)  over the past 10 minutes, per Machine_ID
```

**Type:** `float` — mm/s  
**Source column:** `Vibration_mm_s`  
**Window:** 10-minute time-based rolling window, partitioned by `Machine_ID`  
**Aggregation:** `max`

The highest vibration reading recorded in the last 10 minutes for the same machine. The window slides forward with each new row, so every record carries the worst vibration seen in its recent past rather than just its own instantaneous reading.

Using the rolling maximum instead of the raw value solves a key noise problem: a single outlier reading caused by an external bump or sensor glitch will not persist in the feature beyond one observation, whereas genuine mechanical shock events — loose bearings, drum imbalance, worn mounts — produce elevated readings across many consecutive rows and keep the rolling max high.

**Validation rule:** The service asserts that `Vibration_RollingMax_10min >= Vibration_mm_s` for every row. Any violation is logged as an error.

**Why use it:** Provides the ML model with a robust, noise-resistant real-time mechanical health signal. It catches shock events and sustained vibration anomalies that a point-in-time reading would miss or misrepresent.

---

### `Current_Imbalance_RollingMean_5min` *(streaming feature)*

**Formula:**
```
mean(Current_Imbalance_Ratio)  over the past 5 minutes, per Machine_ID
```

**Type:** `float` — dimensionless ratio  
**Source column:** `Current_Imbalance_Ratio` *(derived, see above)*  
**Window:** 5-minute time-based rolling window, partitioned by `Machine_ID`  
**Aggregation:** `mean`

The average three-phase current imbalance ratio over the last 5 minutes for the same machine. Because `Current_Imbalance_Ratio` is already a normalised, dimensionless scalar, its 5-minute mean is easy to threshold and interpret: a value near 0 means the motor is drawing balanced current; a value trending upward means an electrical fault is developing.

The 5-minute window is deliberately shorter than the vibration window (10 minutes). Current imbalance responds faster to faults, so a tighter window keeps the feature reactive. The rolling mean rather than max is used here because the imbalance signal is already smooth enough — averaging over 5 minutes filters out millisecond-level switching transients without losing the fault trend.

**Validation rule:** The service checks that the rolling mean stays within the global `[min, max]` range of `Current_Imbalance_Ratio`. Values outside that range indicate a computation error and are logged as a warning.

**Why use it:** Electrical faults appear in current imbalance earlier than in vibration or temperature. Pairing this feature with `Vibration_RollingMax_10min` gives the model two complementary real-time channels — electrical and mechanical — which together substantially reduce false-negative anomaly detections compared to either signal in isolation.

---

### `Daily_Vibration_PeakMean_Ratio` *(batch feature)*

**Formula:**
```
max(Vibration_mm_s) / mean(Vibration_mm_s)  per [Machine_ID, calendar day]
```

**Type:** `float` — dimensionless ratio  
**Source column:** `Vibration_mm_s`  
**Aggregation period:** Daily, partitioned by `Machine_ID`  
**Join:** Left-joined back to every row belonging to that machine-day

The ratio of the peak vibration reading to the average vibration reading across the entire calendar day for a given machine. It is computed once per `[Machine_ID, day]` and then broadcast to every individual row in that group, so the ML model receives this daily context alongside the row's real-time readings without a separate lookup.

A low ratio means the machine's vibration was consistent throughout the day — occasional peaks were close to the baseline, suggesting normal operation. A high ratio means there were extreme spikes relative to the daily average, indicating the machine experienced episodes of severe mechanical stress. Because this is computed over hundreds of cycles in a full day, it is statistically stable and not fooled by brief external disturbances.

**Validation rule:** The service checks for null values after the join and logs min/mean/max statistics.

**Why use it:** Gives the model daily-resolution mechanical health context that streaming windows cannot see. Anomaly labels strongly correlate with days where this ratio is persistently above the normal distribution, making it a high-signal feature for distinguishing a one-off bump (low daily ratio) from a machine that is genuinely deteriorating (high daily ratio sustained across the day).

---

### `Weekly_Current_StdDev` *(batch feature)*

**Formula:**
```
stddev(Current_L1)  per [Machine_ID, calendar week]
```

**Type:** `float` — Amperes  
**Source column:** `Current_L1`  
**Aggregation period:** Weekly, partitioned by `Machine_ID`  
**Join:** Left-joined back to every row belonging to that machine-week

The standard deviation of the L1 phase current across the entire calendar week for a given machine. Like all batch features, it is computed once per `[Machine_ID, week]` and then joined to every individual row so the model always has access to it.

A motor with healthy windings and bearings draws current that varies only in response to the wash program being run — structured, predictable variation. As insulation degrades or mechanical resistance in the drum increases, the motor controller compensates erratically and current draw becomes noisier. The within-week standard deviation grows as this degradation progresses. A weekly window smooths out day-to-day load differences (e.g. heavy vs. light wash programs) and isolates the underlying electrical degradation trend. Even with only one month of historical data, the model sees four weekly data points per machine — sufficient to learn the normal range and flag machines whose standard deviation is trending upward.

**Validation rule:** The service checks for null values after the join and logs min/mean/max statistics.

**Why use it:** Captures slow-developing motor degradation that is completely invisible to short rolling windows. It is the only feature in this pipeline designed to detect gradual, multi-day deterioration trends rather than acute events, giving the ML model a long-horizon health signal to complement the real-time streaming features.

---

## Features

The service computes four features across its two pipelines. Each feature is designed to expose a different failure signature of industrial washing machines to the downstream ML model.

---

### Streaming Features (Rolling Windows)

Streaming features are short-term rolling aggregations computed per machine in real time. They are designed to catch anomalies within a single operational cycle or a few minutes of machine operation, giving the model immediate situational awareness.

---

#### `Vibration_RollingMax_10min`

| Property | Value |
|---|---|
| **Source column** | `Vibration_mm_s` |
| **Aggregation** | `max` |
| **Window** | 10 minutes |
| **Pipeline** | Streaming |

The maximum vibration reading observed in the past 10 minutes, computed independently per machine.

A single vibration spike can be caused by something as mundane as an unbalanced load or an external knock and is not reliably informative on its own. By taking the rolling maximum rather than the instantaneous value, the feature stays elevated if the shock was genuine and sustained, while naturally recovering if it was a one-off noise event. This makes the ML model resilient to single-reading outliers without sacrificing sensitivity to real mechanical events like bearing impacts or drum imbalance.

---

#### `Current_Imbalance_RollingMean_5min`

| Property | Value |
|---|---|
| **Source column** | `Current_Imbalance_Ratio` *(derived)*  |
| **Derivation formula** | `(max(L1, L2, L3) − min(L1, L2, L3)) / mean(L1, L2, L3)` |
| **Aggregation** | `mean` |
| **Window** | 5 minutes |
| **Pipeline** | Streaming |

The 5-minute rolling mean of a three-phase current imbalance ratio, computed independently per machine. The imbalance ratio itself is pre-computed as a derived column before the rolling window is applied.

Motor faults — winding degradation, bearing wear, phase loss — manifest as electrical current imbalance *before* they appear in vibration or temperature readings. A healthy motor keeps the imbalance ratio below roughly 0.02; a rising value is an early electrical warning of mechanical trouble. The 5-minute rolling mean filters out millisecond-level switching noise while remaining reactive enough to catch an emerging fault within the same operational cycle.

This feature is intentionally paired with `Vibration_RollingMax_10min`. Together they give the model two complementary real-time signals — one electrical, one mechanical — which dramatically reduces false-negative anomaly detections compared to either signal alone.

---

### Batch Features (Daily / Weekly Aggregations)

Batch features are long-term aggregations computed per `[Machine_ID, period]` and then joined back to every individual row. This means each record carries both its real-time sensor reading and the broader operational context for that machine on that day or week — giving the ML model the full picture without requiring a separate lookup at inference time.

---

#### `Daily_Vibration_PeakMean_Ratio`

| Property | Value |
|---|---|
| **Source column** | `Vibration_mm_s` |
| **Aggregation** | `max(Vibration_mm_s) / mean(Vibration_mm_s)` |
| **Period** | Daily (per `Machine_ID`) |
| **Pipeline** | Batch |

The ratio of peak-to-mean vibration across a full calendar day, computed per machine.

A single vibration spike can be caused by an external bump and carries little information on its own. When the daily peak-to-mean ratio is consistently high, it means the machine experienced repeated or sustained shock events across hundreds of cycles — a much stronger indicator of mechanical deterioration such as a loose bearing or unbalanced drum. Computing this over a full day of data removes the noise inherent in short streaming windows and gives the model a stable daily health score. Anomaly labels typically correlate with days where this ratio sits persistently above the normal distribution, making it a high-signal feature for distinguishing momentary bumps (low daily ratio) from persistent faults (high daily ratio).

---

#### `Weekly_Current_StdDev`

| Property | Value |
|---|---|
| **Source column** | `Current_L1` |
| **Aggregation** | `stddev` |
| **Period** | Weekly (per `Machine_ID`) |
| **Pipeline** | Batch |

The standard deviation of phase-L1 current draw across a full calendar week, computed per machine.

A healthy motor draws current with low within-week variability. As insulation degrades or mechanical resistance in the drum increases, the motor controller compensates by varying the current draw more aggressively — the within-week standard deviation grows. This degradation trend is completely invisible to 5- or 10-minute streaming windows because it develops slowly across many cycles over days. A weekly aggregation smooths out normal load fluctuations caused by different wash program types and exposes the underlying deterioration signal. Even with a single month of data, the model sees four weekly data points per machine — enough to learn the normal range and flag machines whose weekly standard deviation is trending upward.

---

## Configuration Reference

All pipeline behaviour is controlled by `config/feature_engineering_config.yaml`.

### `datasets`

Defines the input/output locations for each dataset to process.

| Field | Type | Description |
|---|---|---|
| `name` | string | Unique dataset identifier (used with `--dataset` CLI flag) |
| `input_path` | string | Path to the source Parquet directory |
| `output_path` | string | Path where enriched Parquet files will be written |
| `file_format` | string | Currently only `"parquet"` is supported |
| `has_labels` | bool | Whether this dataset contains an anomaly label column |
| `label_column` | string | *(Optional)* Name of the label column when `has_labels: true` |

### `schema`

| Field | Type | Description |
|---|---|---|
| `timestamp_column` | string | Name of the timestamp column (cast to `TimestampType` during processing) |
| `partition_columns` | list | Columns used to isolate rolling windows per entity (e.g. `["Machine_ID"]`) |

### `rolling_features` (Streaming Pipeline)

Each entry defines one short-term rolling window feature. Add new features here without changing Python code.

| Field | Type | Description |
|---|---|---|
| `feature_name` | string | Output column name written to Parquet |
| `description` | string | Human-readable explanation of the feature's signal value |
| `source_expression` | string | `"column"` (use `source_column` directly) or `"derived"` (use a pre-computed column from `_compute_derived_columns`) |
| `source_column` | string | Raw column to aggregate *(used when `source_expression: column`)* |
| `derived_column` | string | Name of the pre-computed derived column *(used when `source_expression: derived`)* |
| `aggregation` | string | `max` or `mean` |
| `window_duration` | string | Duration string, e.g. `"10 minutes"`, `"30 seconds"` |
| `window_type` | string | `"time_based"` (currently the only supported type) |
| `enabled` | bool | Set `false` to skip this feature without removing the config entry |

**Currently configured rolling features:** `Vibration_RollingMax_10min`, `Current_Imbalance_RollingMean_5min`. See the [Features](#features) section for full descriptions.

### `batch_features` (Batch Pipeline)

Each entry defines one daily or weekly aggregation joined back to individual rows.

| Field | Type | Description |
|---|---|---|
| `feature_name` | string | Output column name written to Parquet |
| `description` | string | Human-readable explanation |
| `source_column` | string | Raw column to aggregate |
| `aggregation` | string | `ratio_max_mean`, `std`, `mean`, or `max` |
| `aggregation_type` | string | `"daily"` or `"weekly"` |
| `enabled` | bool | Set `false` to skip without removing the config entry |

**Currently configured batch features:** `Daily_Vibration_PeakMean_Ratio`, `Weekly_Current_StdDev`. See the [Features](#features) section for full descriptions.

### `processing`

| Field | Type | Default | Description |
|---|---|---|---|
| `repartition` | bool | `false` | Whether to repartition the DataFrame before writing |
| `num_partitions` | int | `10` | Target partition count when `repartition: true` |
| `cache_intermediate` | bool | `true` | Cache the DataFrame after quality checks to avoid re-reading for both pipelines |
| `write_mode` | string | `"overwrite"` | Spark write mode: `overwrite`, `append`, or `error` |

### `data_quality`

| Field | Type | Description |
|---|---|---|
| `check_null_timestamps` | bool | Filter out rows where the timestamp is null |
| `check_duplicate_timestamps` | bool | Drop duplicate `(timestamp, Machine_ID)` pairs |
| `fill_missing_values` | bool | *(Currently a no-op placeholder for future imputation logic)* |
| `validate_rolling_windows` | bool | After each rolling feature is computed, verify it is numerically consistent (e.g. rolling max ≥ source value) |
| `validate_batch_features` | bool | After each batch feature join, check for null values and log min/max/mean statistics |
| `verify_timestamp_order` | bool | Before writing, log per-machine timestamp ranges and a sample of the final row order |

### `spark_config`

Arbitrary Spark properties passed directly to `SparkSession.builder.config()`. The defaults are tuned for adaptive query execution on medium-sized industrial datasets.

```yaml
spark_config:
  spark.sql.shuffle.partitions: 200
  spark.sql.adaptive.enabled: true
  spark.sql.adaptive.coalescePartitions.enabled: true
```

---

## Output Format

All datasets are written as **flat Parquet** (no `partitionBy` directory structure). Files land as `{output_path}/part-*.parquet` and are sorted by `timestamp → Machine_ID` before writing. This flat layout is intentional — downstream consumers read the full dataset without needing to discover nested partition directories.

---

## Extending the Service

### Adding a new streaming feature

1. Open `config/feature_engineering_config.yaml`.
2. Add a new entry under `rolling_features` following the schema above.
3. If the feature requires a composite input signal, add the derived column computation to `_compute_derived_columns()` in `data_engineering.py` and set `source_expression: "derived"` in the YAML.
4. Set `enabled: true`. No other code changes are required.

### Adding a new batch feature

1. Add a new entry under `batch_features` in the YAML.
2. Choose an existing `aggregation` (`ratio_max_mean`, `std`, `mean`, `max`) or add a new `elif` branch in `_apply_batch_features()`.
3. Set `enabled: true`.

### Supporting a new aggregation type

For rolling features, add a new `elif aggregation == '...'` branch inside the loop in `_apply_rolling_features()`.

For batch features, add a new `elif aggregation == '...'` branch inside the loop in `_apply_batch_features()`.

---

## Datasets

| Dataset | Labels | Description |
|---|---|---|
| `industrial_washer_normal` | No | Baseline sensor readings from healthy machine operation |
| `industrial_washer_with_anomalies` | Yes (`is_anomaly`) | Sensor readings containing labelled fault events for model training |

Input data is read from `data/synthetic_datasets/` and enriched outputs are written to `data/processed_datasets/`.

---

## Logging

The service uses Python's standard `logging` module at `INFO` level. Spark's own log level is set to `WARN` to reduce noise. Key lifecycle events — dataset reads, feature creation, validation results, and write confirmation — are logged with `✓` / `⚠` / `❌` prefixes for easy grepping.
