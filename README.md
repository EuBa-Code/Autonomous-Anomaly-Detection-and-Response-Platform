# 🏭 Industrial IoT Anomaly Detection Pipeline

An end-to-end MLOps pipeline for real-time anomaly detection in industrial washing machines.
Built with **FastAPI**, **Apache Spark**, **Feast**, **Redpanda**, **Quix Streams**, and **Docker Compose**.

## 🚀 Architecture

1.  **Data Generation & Engineering**: Synthetic telemetry data generation and batch feature engineering using **PySpark**.
2.  **Feature Store**: **Feast** serves features for training (Offline - Parquet) and real-time inference (Online - Redis).
3.  **Model Training**: An Isolation Forest model is automatically trained on PySpark's batch data and pushed/versioned in **MLflow**.
4.  **Real-Time Streaming**: **Quix Streams** processes live sensor readings from **Redpanda** (Kafka) to generate rolling-window features.
5.  **Inference Serving**: A real-time HTTP API (**FastAPI**) intercepts data and predicts anomalies using the MLflow model and Redis online features.

## 🛠️ Prerequisites

-   **Docker Desktop** (with at least 4GB RAM allocated).
-   **Make** (optional, but highly recommended for the commands below).

---

## 🏁 Quick Start: The 3-Step Pipeline

The entire system is orchestrated via Docker Compose profiles and `Make` commands to guarantee sequence and stability.

### STEP 0: Clean State (Optional but recommended)
Ensure you have a completely clean state to avoid cache conflicts or old volumes.
```bash
make clean
```

### STEP 1: Offline Setup & Training
Automatically generates the synthetic datasets, runs PySpark to extract features, provisions the Feast registry, and trains the MLflow anomaly detection model.
```bash
make setup
```
> **⏳ Wait for Completion:** This phase takes ~2-3 minutes.
> Run `make logs-setup` and wait until you see: `TRAINING #1 COMPLETATO CON SUCCESSO!`.

### STEP 2: Online Inference & Streaming Services
Once the training is done, spin up the real-time AI agents (FastAPI Inference Server, Feast HTTP Server, Quix Streaming Service).
```bash
make online
```
> You can monitor their startup with `make logs-online`.

### STEP 3: Start the Telemetry Simulation
The system is hungry for live data. Launch the producer to simulate machines sending sensor telemetry to the broker.
```bash
make simulation
```
> Track the data flow with `make logs-simulation`.

---

## 📊 Dashboard & Monitoring

Once everything is running, explore the following interfaces in your browser:

- 🧠 **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000) (Check models and metrics)
- 🐼 **Redpanda Console**: [http://localhost:8080](http://localhost:8080) (Inspect Kafka topics & messages)
- 🍇 **Feast Feature Server**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI for Feast APIs)
- 🔮 **FastAPI Inference**: `http://localhost:8001/docs` (Swagger UI for Predictions - note port depends on inference setup)

## 🛑 Shutting Down

To elegantly stop the cluster without losing data (volumes preserved):
```bash
make stop
```

To totally reset the project and wipe the synthetic data databases and models:
```bash
make clean
```
