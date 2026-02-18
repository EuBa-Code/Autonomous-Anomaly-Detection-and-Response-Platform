# Guida all'uso del Makefile

Il Makefile è stato riorganizzato per seguire il flusso logico del progetto MLOps.

## Comandi Principali

| Comando | Descrizione | Step Pipeline |
| :--- | :--- | :--- |
| `make infra` | Avvia l'infrastruttura di base (MLflow, Redis) | **1** |
| `make data` | Genera i dataset sintetici | **2** |
| `make ingestion` | Esegue l'ingestione dei dati storici (Spark) | **3** |
| `make train` | Esegue il training del modello (MLflow + Sklearn) | **4** |
| **`make pipeline`** | **Esegue l'intera sequenza offline (1 -> 2 -> 3 -> 4)** | **TUTTO** |

## Comandi Real-time

| Comando | Descrizione |
| :--- | :--- |
| `make streaming` | Avvia i servizi di streaming (Redpanda, Quix) e Inference |

## Utility

| Comando | Descrizione |
| :--- | :--- |
| `make stop` | Ferma i container (mantiene i volumi) |
| `make clean` | **RESET COMPLETO**: Ferma container e rimuove volumi e dati |
| `make logs-mlflow` | Mostra i log di MLflow |
| `make logs-train` | Mostra i log del training |
