# Feast Feature Store Service

> Production-ready feature store for the Washing Machine Anomaly Detection System

[![Feast](https://img.shields.io/badge/Feast-0.39+-blue.svg)](https://feast.dev)
[![Redis](https://img.shields.io/badge/Redis-7.0-red.svg)](https://redis.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Feature Definitions](#feature-definitions)
- [API Usage](#api-usage)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The Feast Feature Store service provides a production-grade feature serving layer for real-time anomaly detection in industrial washing machines. It manages 11 sensor and engineered features across multiple machines, enabling:

- **Real-time inference** with sub-100ms latency
- **Point-in-time correct** historical features for training
- **Feature versioning** and lineage tracking
- **Consistent serving** between training and inference

### Key Features

- ✅ **11 sensor features** per washing machine
- ✅ **Online store** (Redis) for real-time serving
- ✅ **Offline store** (Parquet) for training data
- ✅ **HTTP API** for easy integration
- ✅ **Python SDK** for advanced use cases
- ✅ **Push & batch** ingestion support

---

## 🏗️ Architecture


### Feature Store Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   FEAST FEATURE STORE                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Feature Server (HTTP API - Port 8001)                  │   │
│  │  • GET  /health                                         │   │
│  │  • POST /get-online-features                            │   │
│  │  • POST /push                                           │   │
│  │  • POST /materialize                                    │   │
│  └────────────┬─────────────────────────┬──────────────────┘   │
│               │                         │                       │
│    ┌──────────▼──────────┐   ┌─────────▼──────────┐           │
│    │  Online Store       │   │  Offline Store      │           │
│    │  (Redis)            │   │  (Parquet Files)    │           │
│    │                     │   │                     │           │
│    │  • Sub-100ms reads  │   │  • Training data    │           │
│    │  • Real-time serve  │   │  • Point-in-time    │           │
│    │  • TTL: 24 hours    │   │  • TTL: 365 days    │           │
│    └─────────────────────┘   └─────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Registry (SQLite)                                      │   │
│  │  • Entity: machine (Machine_ID: Int64)                  │   │
│  │  • Feature Views: stream (24h), batch (365d)            │   │
│  │  • Feature Service: machine_anomaly_service_v1          │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Training Flow:
  Parquet Files → Offline Store → get_historical_features() → ML Model

Inference Flow:
  Redis → Online Store → get_online_features() → ML Model (Prediction)

Streaming Flow:
  Kafka/Sensors → push() → Online Store → Available for Inference
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+
- uv (Python package manager)
- 4GB RAM minimum
- Redis (included in compose)

### 1. Initial Setup

```bash
# Clone the repository
cd /path/to/project

# Ensure directory structure exists
mkdir -p data/registry
mkdir -p data/synthetic_datasets/industrial_washer_normal
mkdir -p data/processed_datasets

# Start Redis
docker-compose up -d redis
```

### 2. Generate Data (First Time Only)

```bash
# Generate synthetic washing machine data
docker-compose up create_datasets

# Engineer features from raw data
docker-compose up hist_ingestion
```

### 3. Deploy Feature Store

```bash
# Register feature definitions with Feast
docker-compose up feature_store_apply

# Start the Feast feature server
docker-compose up -d feature_store_service

# Load historical features into Redis
docker-compose up feature_loader
```

### 4. Verify Installation

```bash
# Check health
curl http://localhost:8001/health
# Expected: {"status":"healthy"}

# Check registered features
docker exec feature_store_service feast feature-views list
```

### 5. Test with Sample Data

```bash
# Push test data
curl -X POST http://localhost:8001/push \
  -H 'Content-Type: application/json' \
  -d '{
    "push_source_name": "washing_stream_source",
    "df": {
      "Machine_ID": [1],
      "timestamp": ["2024-02-14T10:00:00"],
      "Current_L1": [12.5],
      "Vibration_mm_s": [2.3],
      "Motor_RPM": [1200.0],
      ...
    },
    "to": "online"
  }'

# Get features
curl -X POST http://localhost:8001/get-online-features \
  -H 'Content-Type: application/json' \
  -d '{
    "features": ["machine_stream_features:Current_L1"],
    "entities": {"Machine_ID": [1]}
  }'
```

**🎉 You're ready to go!**

---

## 📁 Project Structure

```
services/feature_store_service/
├── config/
│   └── feature_store.yaml          # Feast configuration
├── src/
│   ├── __init__.py                 # Package exports
│   ├── data_sources.py             # Batch & streaming sources
│   ├── entity.py                   # Machine entity definition
│   ├── features.py                 # Feature view definitions
│   ├── feature_services.py         # Feature service (ML contract)
│   └── test_functionality.py       # Integration tests
├── Dockerfile                      # Container definition
└── README.md                       # This file

# Generated at runtime:
data/
├── registry/
│   └── registry.db                 # Feast metadata store
├── synthetic_datasets/
│   └── industrial_washer_normal/   # Raw data
│       └── machines_batch_features/
└── processed_datasets/
    └── engineered_features/        # Feature-engineered data
```

### File Descriptions

#### Configuration

**`config/feature_store.yaml`**
```yaml
project: anomaly_detection
registry: /feature_store_service/data/registry/registry.db
provider: local

online_store:
  type: redis
  connection_string: "redis:6379"
  key_ttl_seconds: 86400  # 24 hours

offline_store:
  type: file
```

Core Feast configuration defining:
- Project name
- Registry location (metadata)
- Online store (Redis for serving)
- Offline store (Parquet for training)

#### Feature Definitions

**`src/entity.py`** - Entity Definition
```python
machine = Entity(
    name="machine",
    join_keys=["Machine_ID"],
    value_type=ValueType.INT64
)
```

Defines the primary key (`Machine_ID`) for all features.

**`src/data_sources.py`** - Data Sources
- **`machines_batch`**: Historical Parquet files for training
- **`stream_source`**: Real-time push source for inference

**`src/features.py`** - Feature Views
- **`machine_stream_features`**: Real-time features (TTL: 24h)
  - 11 sensor features per machine
- **`machines_batch_features`**: Historical features (TTL: 365d)
  - Same 11 features for training

**`src/feature_services.py`** - Feature Service
- **`machine_anomaly_service_v1`**: Production ML contract
  - Groups all features needed for inference
  - Versioned for safe updates

---

## 🔧 Feature Definitions

### Entity: Machine

| Field | Type | Description |
|-------|------|-------------|
| `Machine_ID` | Int64 | Unique identifier for each washing machine |

### Feature Views

#### machine_stream_features (Real-time)

**TTL:** 24 hours  
**Source:** Push (streaming)  
**Use:** Online inference

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `Cycle_Phase_ID` | Int64 | Current cycle phase (1-4) | - |
| `Current_L1` | Float32 | Phase L1 current | Amps |
| `Current_L2` | Float32 | Phase L2 current | Amps |
| `Current_L3` | Float32 | Phase L3 current | Amps |
| `Voltage_L_L` | Float32 | Line-to-line voltage | Volts |
| `Water_Temp_C` | Float32 | Water temperature | Celsius |
| `Motor_RPM` | Float32 | Motor speed | RPM |
| `Water_Flow_L_min` | Float32 | Water flow rate | L/min |
| `Vibration_mm_s` | Float32 | Current vibration | mm/s |
| `Water_Pressure_Bar` | Float32 | Water pressure | Bar |
| `Vibration_RollingMax_10min` | Float32 | Max vibration (10min window) | mm/s |

#### machines_batch_features (Historical)

**TTL:** 365 days  
**Source:** Parquet files  
**Use:** Training data

Same 11 features as above, but loaded from historical data files.

### Feature Service

**`machine_anomaly_service_v1`**

Groups all 11 features for the anomaly detection model. Provides:
- **Versioning**: Safe updates without breaking consumers
- **Consistency**: Same features for training and serving
- **Simplicity**: Request by service name, not individual features

---

## 💻 API Usage

### HTTP REST API

Base URL: `http://localhost:8001`

#### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{"status": "healthy"}
```

#### 2. Get Online Features

```bash
POST /get-online-features
Content-Type: application/json

{
  "features": [
    "machine_stream_features:Current_L1",
    "machine_stream_features:Vibration_mm_s"
  ],
  "entities": {
    "Machine_ID": [1, 2, 3]
  }
}
```

**Response:**
```json
{
  "metadata": {
    "feature_names": ["Machine_ID", "Current_L1", "Vibration_mm_s"]
  },
  "results": [
    {
      "values": [1, 12.5, 2.3],
      "statuses": ["PRESENT", "PRESENT", "PRESENT"],
      "event_timestamps": ["2024-02-14T10:00:00Z", ...]
    }
  ]
}
```

#### 3. Get Features by Service

```bash
POST /get-online-features

{
  "feature_service": "machine_anomaly_service_v1",
  "entities": {"Machine_ID": [1]}
}
```

Returns all 11 features for Machine 1.

#### 4. Push Streaming Features

```bash
POST /push

{
  "push_source_name": "washing_stream_source",
  "df": {
    "Machine_ID": [1, 2],
    "timestamp": ["2024-02-14T10:00:00", "2024-02-14T10:00:00"],
    "Current_L1": [12.5, 13.2],
    "Vibration_mm_s": [2.3, 2.1],
    ...
  },
  "to": "online"
}
```

#### 5. Materialize Batch Features

```bash
POST /materialize

{
  "start_date": "2024-02-01T00:00:00",
  "end_date": "2024-02-14T23:59:59",
  "feature_views": ["machine_batch_features"]
}
```

### Python SDK

```python
from feast import FeatureStore

# Initialize
store = FeatureStore(repo_path="/feature_store_service")

# Get online features for inference
features = store.get_online_features(
    features=[
        "machine_stream_features:Current_L1",
        "machine_stream_features:Vibration_mm_s"
    ],
    entity_rows=[
        {"Machine_ID": 1},
        {"Machine_ID": 2}
    ]
).to_dict()

# Get historical features for training
training_df = store.get_historical_features(
    entity_df=labels_df,
    features=[
        "machine_batch_features:Current_L1",
        "machine_batch_features:Vibration_mm_s"
    ]
).to_df()

# Push streaming data
import pandas as pd
from datetime import datetime

streaming_data = pd.DataFrame({
    "Machine_ID": [1, 2],
    "timestamp": [datetime.now(), datetime.now()],
    "Current_L1": [12.5, 13.2],
    ...
})

store.push(
    push_source_name="washing_stream_source",
    df=streaming_data,
    to="online"
)
```

### Use Cases

#### Use Case 1: Real-time Anomaly Detection

```python
# In your inference service
features = store.get_online_features(
    feature_service="machine_anomaly_service_v1",
    entity_rows=[{"Machine_ID": machine_id}]
).to_dict()

# Make prediction
prediction = model.predict([features])
```

#### Use Case 2: Training Data Preparation

```python
# Load labels
labels_df = pd.read_parquet("labels.parquet")

# Get point-in-time correct features
training_df = store.get_historical_features(
    entity_df=labels_df,
    feature_service="machine_anomaly_service_v1"
).to_df()

# Train model
X = training_df.drop(['label'], axis=1)
y = training_df['label']
model.fit(X, y)
```

#### Use Case 3: Streaming Pipeline Integration

```python
# In your Kafka consumer
from kafka import KafkaConsumer

consumer = KafkaConsumer('machine-telemetry')

for message in consumer:
    data = parse_message(message)
    df = pd.DataFrame([data])
    
    # Push to Feast for immediate serving
    store.push(
        push_source_name="washing_stream_source",
        df=df,
        to="online"
    )
```

---

## 🛠️ Development

### Adding New Features

1. **Define the feature in `features.py`:**

```python
machine_stream_features = FeatureView(
    name="machine_stream_features",
    entities=[machine],
    ttl=timedelta(hours=24),
    schema=[
        # Existing features...
        Field(name="New_Feature", dtype=Float32),  # Add here
    ],
    source=stream_source,
)
```

2. **Apply the changes:**

```bash
docker-compose up feature_store_apply
```

3. **Restart the server:**

```bash
docker-compose restart feature_store_service
```

### Running Tests

```bash
# Run integration tests
docker exec -it feature_store_service python src/test_functionality.py

# Test specific feature retrieval
docker exec -it feature_store_service python -c "
from feast import FeatureStore
store = FeatureStore(repo_path='/feature_store_service')
print(store.get_online_features(
    features=['machine_stream_features:Current_L1'],
    entity_rows=[{'Machine_ID': 1}]
).to_dict())
"
```

### Debugging

```bash
# View logs
docker logs feature_store_service -f

# Check Redis data
docker exec -it redis redis-cli
> KEYS *
> GET <key>

# Inspect registry
docker exec -it feature_store_service sqlite3 data/registry/registry.db
> .tables
> SELECT * FROM entities;
```

---

## 🚢 Deployment

### Docker Compose (Current Setup)

Add to your `compose.yaml`:

```yaml
services:
  # Feature Store Apply - Run once to register features
  feature_store_apply:
    build: 
      context: .
      dockerfile: services/feature_store_service/Dockerfile
    container_name: feature_store_apply
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./data/registry:/feature_store_service/data/registry
      - ./data/synthetic_datasets/industrial_washer_normal:/feature_store_service/data/offline:ro
    networks:
      - anomaly-detection-network
    restart: 'no'

  # Feature Server - HTTP API
  feature_store_service:
    build: 
      context: .
      dockerfile: services/feature_store_service/Dockerfile
    container_name: feature_store_service
    depends_on:
      redis:
        condition: service_healthy
      feature_store_apply:
        condition: service_completed_successfully
    environment:
      - FEAST_REPO_PATH=/feature_store_service
    volumes:
      - ./data/registry:/feature_store_service/data/registry:ro
      - ./data/synthetic_datasets/industrial_washer_normal:/feature_store_service/data/offline:ro
    ports:
      - '8001:6566'
    networks:
      - anomaly-detection-network
    restart: unless-stopped
    command: ["uv", "run", "feast", "serve", "-h", "0.0.0.0"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6566/health"]
      interval: 10s
      timeout: 3s
      retries: 3
```

### Production Considerations

#### 1. High Availability

- **Redis Sentinel** for automatic failover
- **Redis Cluster** for horizontal scaling
- **Load Balancer** across multiple feature server instances

#### 2. Monitoring

```python
# Add metrics endpoint
from prometheus_client import Counter, Histogram

feature_requests = Counter('feast_feature_requests', 'Feature requests')
feature_latency = Histogram('feast_feature_latency', 'Feature latency')
```

#### 3. Security

- **Authentication**: Add API keys or OAuth
- **Network**: Use VPC/private networking
- **Encryption**: Enable Redis TLS

#### 4. Scaling

| Component | Scale Method | Notes |
|-----------|--------------|-------|
| Feature Server | Horizontal | Stateless, add more pods/containers |
| Redis | Vertical | 16GB RAM recommended for 1000+ machines |
| Registry | No scaling needed | SQLite sufficient for metadata |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. FeatureNameCollisionError

**Error:**
```
FeatureNameCollisionError: Duplicate features named Current_L1, ...
```

**Solution:**
Update `feature_services.py` to include only one feature view:

```python
machine_feature_service_v1 = FeatureService(
    name="machine_anomaly_service_v1",
    features=[
        machine_stream_features,  # Remove machines_batch_features
    ]
)
```

Or use `full_feature_names=true` in API requests.

#### 2. Empty Feature Values

**Symptom:** Features return `null` or `NOT_FOUND`

**Solution:**
```bash
# Load data into Redis first
docker-compose up feature_loader

# Or push test data
curl -X POST http://localhost:8001/push ...
```

#### 3. Connection Refused

**Symptom:** Can't connect to `localhost:8001`

**Check:**
```bash
# Is service running?
docker ps | grep feature_store_service

# Check logs
docker logs feature_store_service

# Test from inside container
docker exec feature_store_service curl localhost:6566/health
```

#### 4. Registry Not Found

**Error:**
```
FileNotFoundError: registry.db not found
```

**Solution:**
```bash
# Run feast apply first
docker-compose up feature_store_apply

# Verify registry exists
ls -la data/registry/registry.db
```

### Performance Tuning

#### Redis Configuration

```redis
# In redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for speed
```

#### Feature Server

```yaml
# Increase workers
command: ["uv", "run", "feast", "serve", "-h", "0.0.0.0", "--workers", "4"]
```

### Logging

```bash
# Enable debug logging
docker-compose up feature_store_service -e LOG_LEVEL=DEBUG

# View detailed logs
docker logs feature_store_service --tail 100 -f
```

---

## 📚 Additional Resources

### Documentation

- [Feast Official Docs](https://docs.feast.dev)
- [Redis Documentation](https://redis.io/docs)
- [Feature Store Concepts](https://docs.feast.dev/getting-started/concepts)

### Related Services

- **create_datasets**: Generates synthetic washing machine sensor data
- **hist_ingestion**: Feature engineering pipeline (Spark)
- **feature_loader**: Loads historical features to Redis
- **training_service**: Model training using historical features
- **ingestion_service**: Real-time feature streaming

### API Reference

- **Feast Python SDK**: [GitHub](https://github.com/feast-dev/feast)
- **Feature Server API**: [Swagger Docs](http://localhost:8001/docs)

---

## 🤝 Contributing

### Adding a New Feature

1. Update `features.py` with new field
2. Run `docker-compose up feature_store_apply`
3. Update documentation
4. Test with sample data

### Running Tests

```bash
make test-feast  # Run all tests
make test-api    # Test API endpoints
make test-sdk    # Test Python SDK
```

---

## 📄 License

[Your License Here]

---

## 👥 Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting)
- Review [Feast Docs](https://docs.feast.dev)
- Open an issue in the repository

---

**Built with ❤️ for production ML systems**
