# Feast Feature Store Service

> Production-ready feature store for the Washing Machine Anomaly Detection System

[![Feast](https://img.shields.io/badge/Feast-0.39+-blue.svg)](https://feast.dev)
[![Redis](https://img.shields.io/badge/Redis-7.0-red.svg)](https://redis.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Data Flow](#data-flow)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Feature Definitions](#-feature-definitions)
- [Additional resources](#-additional-resources)


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
│                   FEAST FEATURE STORE                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Feature Server (HTTP API - Port 8001)                  │   │
│  │  • GET  /health                                         │   │
│  │  • POST /get-online-features                            │   │
│  │  • POST /push                                           │   │
│  │  • POST /materialize                                    │   │
│  └────────────┬─────────────────────────┬──────────────────┘   │
│               │                         │                      │
│    ┌──────────▼──────────┐   ┌─────────▼──────────┐            │
│    │  Online Store       │   │  Offline Store      │           │
│    │  (Redis)            │   │  (Parquet Files)    │           │
│    │                     │   │                     │           │
│    │  • Sub-100ms reads  │   │  • Training data    │           │
│    │  • Real-time serve  │   │  • Point-in-time    │           │
│    │  • TTL: 24 hours    │   │  • TTL: 365 days    │           │
│    └─────────────────────┘   └─────────────────────┘           │
│                                                                │
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


```bash
make run_feature_store
```

### 2. TEST with POSTMAN (First Time Only)

```bash
import in postman
```

[FILE](\Feast_API_Tests.postman_collection.json)


**🎉 You're ready to go!**

---

## 📁 Project Structure

```
feature_store_service/
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
    └── registry.db                 

```



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



## 📚 Additional Resources

### Documentation

- [Feast Official Docs](https://docs.feast.dev)
- [Redis Documentation](https://redis.io/docs)
- [Feature Store Concepts](https://docs.feast.dev/getting-started/concepts)


## 🤝 Contributing

### Adding a New Feature

1. Update `features.py` with new field
2. Run `make run_feature_store`
3. Update documentation
4. Test with sample data

---

**Built with ❤️ for production ML systems**
