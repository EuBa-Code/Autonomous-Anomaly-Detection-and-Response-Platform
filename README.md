# 🏭 Industrial IoT Anomaly Detection Pipeline

An end-to-end MLOps pipeline for real-time anomaly detection in industrial washing machines.
Built with **FastAPI**, **Apache Spark**, **Feast**, **Redpanda**, **Quix Streams**, and **Docker Compose**.

## 🚀 Architecture

1.  **Data Generation**: Synthetic telemetry data (Vibration, Temperature, etc.) generated via Spark.
2.  **Feature Engineering**: Batch processing (Spark) for historical features, Real-time processing (Quix) for streaming features.
3.  **Feature Store**: **Feast** serves features for training (Offline - Parquet) and inference (Online - Redis).
4.  **Training**: Isolation Forest model trained on historical data and tracked/versioned in **MLflow**.
5.  **Inference**: Real-time HTTP API (FastAPI) predicts anomalies using the trained model (from MLflow) and online features.

## 🛠️ Prerequisites

-   **Docker Desktop** (with at least 4GB RAM allocated).
-   **Make** (optional, but recommended for easy commands).
-   **Curl** or **Postman** for testing.

## 🏁 Quick Start

### 1. Clean Environment
Ensure a clean state to avoid conflicts:
```bash
make clean
```

### 2. Run Offline Pipeline (Batch)
Starts infrastructure (MLflow, Redis), generates data, ingests historical features, and trains the model.
```bash
make pipeline
```
> **Note:** This step may take a few minutes. Check progress with `make logs-train`.

### 3. Start Real-Time Services
Launches Redpanda, Streaming Service, and Inference API.
```bash
make streaming
```

### 4. Start Simulation (Data Producer)
Begins streaming telemetry data to the system.
```bash
make simulate
```
> **Tip:** Open a separate terminal to run this command, or check logs with `docker logs -f producer_service`.

## 📊 Monitoring
- **MLflow UI**: [http://localhost:5000](http://localhost:5000) (Experiments & Models)
- **Redpanda Console**: [http://localhost:8080](http://localhost:8080) (Topics & Messages)
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Prediction API)

**Expected Response:**
```json
{
  "machine_id": 1,
  "is_anomaly": 0,
  "anomaly_score": -0.1234,
  "model_version": "v1"
}
```

## 📂 Project Structure

-   `services/`: Microservices source code (Inference, Streaming, Feature Store, etc.).
-   `data/`: Generated datasets and model artifacts (ignored in Git).
-   `compose.yaml`: Docker Compose configuration.
-   `makefile`: Shortcut commands.

## 🐛 Troubleshooting

-   **OOM Kill (Exit 137)**: Increase Docker RAM or check `SPARK_DRIVER_MEMORY` in `compose.yaml`.
-   **Redpanda Disk Full**: Run `make prune` to clear old volumes.
-   **Missing Features**: Ensure `make simulate` is running and the Producer is active.
