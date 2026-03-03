# ==============================================================================
# WASHING MACHINES ANOMALY DETECTION - PIPELINE COMMANDS (NEW APPROACH)
# Orchestrated beautifully with Docker Compose profiles
# ==============================================================================
.PHONY: help-new setup online simulation stop clean logs-setup logs-online logs-simulation

help-new:
	@echo "=========================================================================="
	@echo "WASHING MACHINES ANOMALY DETECTION - NEW 3-STEP PIPELINE"
	@echo "=========================================================================="
	@echo "1) make setup        : (STEP 1) Run offline pipeline (Data Gen, PySpark, Feast, Training)"
	@echo "2) make online       : (STEP 2) Run real-time inference (Feast Server, Quix, FastAPI)"
	@echo "3) make simulation   : (STEP 3) Start the Producer to simulate live telemetry sensors"
	@echo "4) make stop         : Stop all running containers gracefully"
	@echo "5) make clean        : Destructive clean (stops containers AND removes all volumes/data)"
	@echo ""
	@echo "Logs:"
	@echo "make logs-setup      : Follow logs for the offline setup phase"
	@echo "make logs-online     : Follow logs for the real-time services"
	@echo "make logs-simulation : Follow logs for the data producer"
	@echo "=========================================================================="

setup:
	@echo "--- [1/3] Starting Offline Setup Pipeline ---"
	@echo "This will generate data, engineer features, and train the MLflow model."
	docker compose --profile setup up -d --build

online:
	@echo "--- [2/3] Starting Online Real-Time Pipeline ---"
	@echo "This will start the FastAPI inference server and Quix streams."
	docker compose up redpanda redpanda-console
	docker compose --profile online up -d --build

simulation:
	@echo "--- [3/3] Starting Telemetry Data Simulation ---"
	@echo "This starts the producer sending fake telemetry through Redpanda."
	docker compose --profile simulation up -d --build

stop:
	@echo "Stopping all services..."
	docker compose --profile setup --profile online --profile simulation down

clean:
	@echo "🧹 Deep cleaning... Removing containers, networks, and all data volumes"
	docker compose --profile setup --profile online --profile simulation down -v
	@echo "Environment completely reset."

logs-setup:
	docker compose --profile setup logs -f

logs-online:
	docker compose --profile online logs -f

logs-simulation:
	docker compose --profile simulation logs -f


# ==============================================================================
# LEGACY COMMANDS (PREVIOUS ITERATIONS)
# Kept for backward compatibility and granular testing
# ==============================================================================

# ----------------------------------------------
# Variables & Configuration
# ----------------------------------------------
.PHONY: help create_datasets data_engineering final_datasets \
        apply_feature_store service_feature_store run_feature_store \
        build_mlflow build_training build_all \
        mlflow_up training_up all_up \
        mlflow_down training_down all_down \
        mlflow_logs training_logs all_logs \
        run_all \
		all

# ----------------------------------------------
# Display help message
# ----------------------------------------------
help:
	@echo "=============================================="
	@echo "DATA PIPELINE & FEATURE STORE MAKEFILE (LEGACY)"
	@echo "=============================================="
	@echo ">> Use 'make help-new' to view the modern 3-step pipeline commands! <<"
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
	@echo "11) clean-legacy           : Stop all containers and remove volumes"
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

run_feature_store: apply_feature_store service_feature_store

# ----------------------------------------------
# Build Services
# ----------------------------------------------
build_mlflow:
	@echo "Building MLflow..."
	docker compose build mlflow

build_training:
	@echo "Building Training Pipeline..."
	docker compose build training_service

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
	docker compose up -d training_service

all_up: mlflow_up training_up
	@echo "All services are up."

mlflow_down:
	@echo "Stopping MLflow..."
	docker compose stop mlflow

training_down:
	@echo "Stopping Training Pipeline..."
	docker compose stop training_service

all_down: mlflow_down training_down
	@echo "All services stopped."

# ----------------------------------------------
# Logs & Maintenance
# ----------------------------------------------
mlflow_logs:
	docker compose logs -f mlflow

training_logs:
	docker compose logs -f training_service

all_logs:
	docker compose logs -f

run_all: final_datasets run_feature_store

clean-legacy:
	@echo "Cleaning up: stopping containers and removing volumes..."
	docker compose down -v
	@echo "Environment cleaned."

all: final_datasets run_feature_store build_mlflow build_training mlflow_up training_up
	@echo "All services are up and running."
	
# ------------------------------------------------------------------------------
# INFRA & OLD PIPELINE COMMANDS
# ------------------------------------------------------------------------------
infra:
	@echo "--- [1/4] Starting Infrastructure (MLflow & Redis) ---"
	docker compose up -d --build mlflow redis
	@echo "Waiting for services to be ready..."
	@timeout /t 10 >nul 2>&1 || ping -n 11 127.0.0.1 >nul

data:
	@echo "--- [2/4] Generating Synthetic Data ---"
	docker compose up --build create_datasets

ingestion:
	@echo "--- [3/4] Ingesting Historical Data ---"
	docker compose up --build hist_ingestion

train:
	@echo "--- [4/4] Training Model ---"
	docker compose up --build training_service

pipeline: infra data ingestion train
	@echo "✅ OFFLINE PIPELINE COMPLETED SUCCESSFULLY"

streaming:
	@echo "--- Starting Real-time Streaming & Inference ---"
	docker compose up -d --build streaming_service inference_service

logs-mlflow:
	docker logs -f mlflow

logs-train:
	docker logs -f training_service

# ---------

# First Training
debug_training:
	docker compose up --build mlflow training_service

offline_files:
	uv run -m utils.create_offline_files

debug_batch:
	docker compose up --build batch_feature_pipeline

debug_inference:
	docker compose up --build inference_service

debug_inference_2:
	docker compose build --no-cache inference_service


debug_all:
	uv run -m utils.create_offline_files && \
	docker compose up --build \
		redpanda \
		redpanda-console \
		redis \
		batch_feature_pipeline \
		feature_store_apply \
		feast_materialize \
		feature_store_service \
		streaming_service \
		producer_service \
		-d
debug_streaming:
		uv run -m utils.create_offline_files && \
		docker compose up --build redpanda redis redpanda-console feature_store_apply feature_store_service batch_feature_pipeline feast_materialize streaming_service producer_service -d
