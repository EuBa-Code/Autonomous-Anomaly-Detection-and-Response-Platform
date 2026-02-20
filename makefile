# ----------------------------------------------
# Variables & Configuration
# ----------------------------------------------
.PHONY: help create_datasets data_engineering final_datasets \
        apply_feature_store service_feature_store run_feature_store \
        build_mlflow build_training build_all \
        mlflow_up training_up all_up \
        mlflow_down training_down all_down \
        mlflow_logs training_logs all_logs \
        run_all clean

# ----------------------------------------------
# Display help message
# ----------------------------------------------
help:
	@echo "=============================================="
	@echo "DATA PIPELINE & FEATURE STORE MAKEFILE"
	@echo "=============================================="
	@echo "1) create_datasets         : Build/Run dataset creation"
	@echo "2) data_engineering        : Run data engineering pipeline"
	@echo "3) final_datasets          : Run both creation & engineering"
	@echo "4) apply_feature_store     : Create Parquet files & Apply Feast Schema"
	@echo "5) service_feature_store   : Run Feast service"
	@echo "6) run_feature_store       : Full Stack (Redis + Apply + Serve)"
	@echo "7) run_all                 : Dataset creation + feature store"
	@echo "8) all_up                  : Start MLflow and Training Pipeline"
	@echo "9) all_down                : Stop MLflow and Training Pipeline"
	@echo "10) all_logs               : View all logs"
	@echo "11) clean                  : Stop all containers and remove volumes"
	@echo "=============================================="

# ----------------------------------------------
# Internal Helpers
# ----------------------------------------------
sync:
	@echo "Syncing environment with uv..."
	@uv sync

# ----------------------------------------------
# Data Pipeline
# ----------------------------------------------
create_datasets: sync
	@echo "Building and running datasets..."
	docker compose up --build create_datasets

data_engineering: sync
	@echo "Building and running data engineering..."
	docker compose up --build data_engineering

final_datasets: create_datasets data_engineering

# ----------------------------------------------
# Feature Store Management
# ----------------------------------------------
apply_feature_store: sync
	@echo "Creating offline storage files..."
	uv run -m --group data-offline utils.create_offline_files
	@echo "Starting infra and applying Feast registry..."
	docker compose up -d redis redpanda
	docker compose up --build --abort-on-container-exit feature_store_apply

service_feature_store:
	@echo "Starting Feast Server..."
	docker compose up --build -d feature_store_service
	@echo "Feast Server is running at http://localhost:6566/health"
	docker compose logs -f feature_store_service

run_feature_store: apply_feature_store service_feature_store

# ----------------------------------------------
# Build Services
# ----------------------------------------------
build_mlflow:
	@echo "Building MLflow..."
	docker compose build mlflow

build_training:
	@echo "Building Training Pipeline..."
	docker compose build training_pipeline

build_all: build_mlflow build_training

# ----------------------------------------------
# Execution (UP / DOWN)
# ----------------------------------------------
mlflow_up:
	@echo "Starting MLflow..."
	docker compose up -d mlflow
	@echo "Waiting for MLflow to be ready..."
	@powershell -Command "Start-Sleep -s 10"

training_up:
	@echo "Starting Training Pipeline..."
	docker compose up -d training_pipeline

all_up: mlflow_up training_up
	@echo "All services are up."

mlflow_down:
	@echo "Stopping MLflow..."
	docker compose stop mlflow

training_down:
	@echo "Stopping Training Pipeline..."
	docker compose stop training_pipeline

all_down: mlflow_down training_down
	@echo "All services stopped."

# ----------------------------------------------
# Logs & Maintenance
# ----------------------------------------------
mlflow_logs:
	docker compose logs -f mlflow

training_logs:
	docker compose logs -f training_pipeline

all_logs:
	docker compose logs -f

run_all: final_datasets run_feature_store

clean:
	@echo "Cleaning up: stopping containers and removing volumes..."
	docker compose down -v
	@echo "Environment cleaned."

all: run_all build_all all_up