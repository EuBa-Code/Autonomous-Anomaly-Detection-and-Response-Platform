# ==============================================================================
# Washing Machines Anomaly Detection - Makefile
# ==============================================================================

.PHONY: help infra data ingestion train pipeline streaming stop clean logs-mlflow logs-train

# Default target
help:
	@echo "Available commands:"
	@echo "  make infra       - Start core infrastructure (MLflow, Redis)"
	@echo "  make data        - Generate synthetic datasets"
	@echo "  make ingestion   - Ingest historical data"
	@echo "  make train       - Train the model"
	@echo "  make pipeline    - Run the full offline pipeline (infra -> data -> ingestion -> train)"
	@echo "  make streaming   - Start real-time streaming services"
	@echo "  make stop        - Stop all services"
	@echo "  make clean       - Stop services and remove volumes (RESET)"

# ------------------------------------------------------------------------------
# 1. CORE INFRASTRUCTURE
# ------------------------------------------------------------------------------
infra:
	@echo "--- [1/4] Starting Infrastructure (MLflow & Redis) ---"
	docker compose up -d --build mlflow redis
	@echo "Waiting for services to be ready..."
	@timeout /t 10 >nul 2>&1 || ping -n 11 127.0.0.1 >nul

# ------------------------------------------------------------------------------
# 2. OFFLINE PIPELINE (Data & Training)
# ------------------------------------------------------------------------------
data:
	@echo "--- [2/4] Generating Synthetic Data ---"
	docker compose up --build create_datasets

ingestion:
	@echo "--- [3/4] Ingesting Historical Data ---"
	docker compose up --build hist_ingestion

train:
	@echo "--- [4/4] Training Model ---"
	docker compose up --build training_service

# Esegue l'intera pipeline sequenziale
pipeline: infra data ingestion train
	@echo "✅ OFFLINE PIPELINE COMPLETED SUCCESSFULLY"

# ------------------------------------------------------------------------------
# 3. ONLINE PIPELINE (Real-time)
# ------------------------------------------------------------------------------
streaming:
	@echo "--- Starting Real-time Streaming & Inference ---"
	docker compose up -d --build streaming_service inference_service

# ------------------------------------------------------------------------------
# 4. UTILITIES
# ------------------------------------------------------------------------------
stop:
	@echo "Stopping services..."
	docker compose down --remove-orphans

clean:
	@echo "Cleaning up (Removing volumes & orphans)..."
	docker compose down -v --remove-orphans
	@echo "Done."

logs-mlflow:
	docker logs -f mlflow

logs-train:
	docker logs -f training_service
