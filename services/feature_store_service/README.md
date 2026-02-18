# Feature Store — API Test Suite

A complete test suite to verify every aspect of your Feature Store (`anomaly_detection` project).
Tests are organized in logical order: from system health, to data ingestion, to feature retrieval.

---

## Architecture Overview

| Service | Description | External Port |
|---|---|---|
| `feature_store_apply` | Runs `feast apply` once to register all feature definitions | — |
| `feature_store_service` | Serves features over HTTP (`feast serve`) | **8000** → internal 6566 |
| `redis` | Online store (key TTL configured in `feature_store.yaml`) | 6379 |
| `redpanda` | Kafka-compatible message broker | 19092 |

> **Base URL for all API calls:** `http://localhost:8000`

---

## Feature Views at a Glance

### `machine_streaming_features` — TTL: 12 hours
Populated by the real-time pipeline (Quixstreams → `washing_stream_push` PushSource).

| Field | Type | Description |
|---|---|---|
| `Cycle_Phase_ID` | Int64 | Current wash cycle phase |
| `Current_L1/L2/L3` | Float32 | Phase currents |
| `Voltage_L_L` | Float32 | Line-to-line voltage |
| `Water_Temp_C` | Float32 | Water temperature |
| `Motor_RPM` | Float32 | Motor speed |
| `Water_Flow_L_min` | Float32 | Water flow rate |
| `Vibration_mm_s` | Float32 | Vibration level |
| `Water_Pressure_Bar` | Float32 | Water pressure |
| `Current_Imbalance_Ratio` | Float32 | 3-phase imbalance scalar |
| `Vibration_RollingMax_10min` | Float32 | 10-min rolling max of vibration |
| `Current_Imbalance_RollingMean_5min` | Float32 | 5-min rolling mean of imbalance ratio |

### `machine_batch_features` — TTL: 7 days
Populated by the batch pipeline (PySpark → partitioned Parquet at `/data/offline/machines_batch_features`).

| Field | Type | Description |
|---|---|---|
| `Daily_Vibration_PeakMean_Ratio` | Float32 | max/mean vibration per machine per day |
| `Weekly_Current_StdDev` | Float32 | Stddev of L1 current per machine per week |

---

## Test Suite

### 1. Connection Test (Health Check)

Verifies that the Feast server is running and can communicate with the Registry (`/data/registry/registry.db`).

- **Method:** `GET`
- **URL:** `http://localhost:8000/health`
- **Expected result:** `200 OK`. If it fails, check that `feature_store_apply` completed successfully before `feature_store_service` started.

---

### 2. Streaming Ingestion Test (Push API)

Simulates sending real-time data as Quixstreams would. Tests whether the `washing_stream_push` PushSource accepts data and writes it to Redis.

- **Method:** `POST`
- **URL:** `http://localhost:8000/push`
- **Body (JSON):**

```json
{
  "push_source_name": "washing_stream_push",
  "df": {
    "Machine_ID": [1001],
    "timestamp": ["2026-02-18T20:00:00Z"],
    "Cycle_Phase_ID": [3],
    "Current_L1": [12.5],
    "Current_L2": [11.8],
    "Current_L3": [12.1],
    "Voltage_L_L": [400.0],
    "Water_Temp_C": [60.0],
    "Motor_RPM": [850.0],
    "Water_Flow_L_min": [18.5],
    "Vibration_mm_s": [2.1],
    "Water_Pressure_Bar": [3.4],
    "Current_Imbalance_Ratio": [0.03],
    "Vibration_RollingMax_10min": [2.8],
    "Current_Imbalance_RollingMean_5min": [0.025]
  },
  "to": "online"
}
```

- **Expected result:** Empty response `{}` with status `200`. The data is now stored in Redis with a **12-hour TTL**.

---

### 3. Online Feature Retrieval Test (Single Entity)

Verifies that Feast retrieves the data just pushed to Redis via the `machine_anomaly_service_v1` FeatureService.

- **Method:** `POST`
- **URL:** `http://localhost:8000/get-online-features`
- **Body (JSON):**

```json
{
  "feature_service": "machine_anomaly_service_v1",
  "entities": {
    "Machine_ID": [1001]
  }
}
```

- **Expected result:** A JSON response containing the values pushed in Test #2. The batch features `Daily_Vibration_PeakMean_Ratio` and `Weekly_Current_StdDev` will be `null` until Parquet materialization is run.

---

### 4. Multi-Entity Retrieval Test (Batch Retrieval)

Verifies that Feast correctly handles requests for multiple machines simultaneously.

- **Method:** `POST`
- **URL:** `http://localhost:8000/get-online-features`
- **Body (JSON):**

```json
{
  "feature_service": "machine_anomaly_service_v1",
  "entities": {
    "Machine_ID": [1001, 9999]
  }
}
```

- **Expected result:** Two result sets. Machine `1001` returns the pushed data; machine `9999` (which does not exist) returns all `null` values with status `NOT_FOUND`.

---

### 5. Type Validation Test (Error Test)

Tests Feast's robustness by sending an incorrect data type — a string where a `Float32` is expected.

- **Method:** `POST`
- **URL:** `http://localhost:8000/push`
- **Body (JSON):**

```json
{
  "push_source_name": "washing_stream_push",
  "df": {
    "Machine_ID": [1001],
    "timestamp": ["2026-02-18T20:00:00Z"],
    "Motor_RPM": ["FAST"]
  },
  "to": "online"
}
```

- **Expected result:** A `400` or `500` error with a message indicating that the string could not be converted to `Float32`.

---

### 6. Expiration Test (TTL — Time To Live)

The `machine_streaming_features` view has a `ttl` of **12 hours** (defined in `features.py`). The Redis key TTL is set to **86400 seconds (24 hours)** in `feature_store.yaml`.

- **How to test:** Push data with a timestamp older than 12 hours (e.g., `2025-01-01T10:00:00Z`), then attempt to read it using Test #3.
- **Push body:** Use the same body as Test #2, replacing the timestamp with `"2025-01-01T10:00:00Z"`.
- **Expected result:** Status `OUTSIDE_MAX_AGE` or `null` values for all streaming features.

---

## Next Steps

Once all tests pass, run **Batch Materialization** to populate the `machine_batch_features` view and eliminate the `null` values for `Daily_Vibration_PeakMean_Ratio` and `Weekly_Current_StdDev`:

```bash
# Inside the feature_store_service container
uv run feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

This reads from the partitioned Parquet files at `/data/offline/machines_batch_features` and loads the computed aggregations into Redis.