.PHONY: help init_linux full-datasets infrastructure airflow feature-store-setup first-training online-services cold-start full-data-flow first-run online-run clean-data stop stop-online clean clean-all rebuild health logs logs-service

# Default target
.DEFAULT_GOAL := help

# Colors for output
YELLOW := \033[1;33m
GREEN := \033[1;32m
RED := \033[1;31m
BLUE := \033[1;34m
NC := \033[0m # No Color

##@ Initialization

init_linux: ## Create required directories with proper permissions
	@echo "$(BLUE)📁 Creating directories...$(NC)"
	sudo mkdir -p redpanda_storage logs data/registry data/offline data/entity_df qdrant_data local_models outputs
	sudo chmod -R 777 redpanda_storage logs data qdrant_data local_models outputs
	@echo "$(GREEN)✅ Directories created$(NC)"

##@ Setup & Data Preparation

full-datasets: init_linux ## Build and create all datasets (synthetic data)
	@echo "$(BLUE)📊 Creating synthetic datasets...$(NC)"
	docker compose up --build \
		create_datasets \
		data_engineering \
		create_offline_files
	@echo "$(GREEN)✅ Datasets created$(NC)"

##@ Infrastructure

infrastructure: ## Start core infrastructure (Redis, Redpanda, MLflow, MongoDB, Qdrant)
	@echo "$(BLUE)🏗️  Starting core infrastructure...$(NC)"
	docker compose up -d \
		redis \
		redis-insight \
		redpanda \
		redpanda-console \
		redpanda-init \
		mlflow \
		qdrant \
		mongodb
	@echo "$(GREEN)✅ Infrastructure started$(NC)"
	@echo "$(YELLOW)📍 Access points:$(NC)"
	@echo "  - Redis Insight:     http://localhost:5540"
	@echo "  - Redpanda Console:  http://localhost:8080"
	@echo "  - MLflow:            http://localhost:5000"
	@echo "  - Qdrant:            http://localhost:6333"
	@echo "  - MongoDB (compass): mongodb://admin:admin@localhost:27017/"

##@ Orchestration

airflow: infrastructure ## Start Airflow for orchestration
	@echo "$(BLUE)🌬️  Starting Airflow...$(NC)"
	docker compose up -d \
		postgres_airflow \
		airflow-init
	@echo "$(YELLOW)⏳ Waiting for Airflow init...$(NC)"
	docker compose wait airflow-init
	docker compose up -d \
		airflow-webserver \
		airflow-scheduler
	@echo "$(GREEN)✅ Airflow started$(NC)"
	@echo "$(YELLOW)📍 Airflow Web UI: http://localhost:8081$(NC)"
	@echo "   Username: airflow"
	@echo "   Password: airflow"

##@ Feature Store Setup

feature-store-setup: infrastructure ## Register features in Feast (one-time setup)
	@echo "$(BLUE)🗄️  Registering features...$(NC)"
	docker compose --profile setup up \
		feature_store_apply
	@echo "$(GREEN)✅ Features registered$(NC)"

##@ Training

first-training: infrastructure ## Run initial model training (requires datasets)
	@echo "$(BLUE)🤖 Training initial model...$(NC)"
	docker compose --profile setup up --build --abort-on-container-exit \
		training_service
	@echo "$(GREEN)✅ Training completed$(NC)"

##@ Online Services

online-services: infrastructure airflow ## Start all online services
	@echo "$(BLUE)🚀 Starting online services...$(NC)"
	docker compose --profile online up -d \
		feature_store_service \
		streaming_service \
		inference_service \
		ingestion_rag \
		mcp_server \
		vllm \
		langchain_service \
		if_anomaly
	@echo "$(GREEN)✅ Online services started$(NC)"
	@echo "$(YELLOW)📍 Service endpoints:$(NC)"
	@echo "  - Feature Store:     http://localhost:8000"
	@echo "  - Inference API:     http://localhost:8001"
	@echo "  - MCP Server:        http://localhost:8020"
	@echo "  - vLLM:              http://localhost:8222"
	@echo "  - LangChain Agent:   http://localhost:8010"

##@ Cold Start (first materialization)

cold-start: ## Populate Redis with historical features
	@echo "$(BLUE)❄️  Running cold start...$(NC)"
	docker compose --profile online up --build \
		cold_start
	@echo "$(GREEN)✅ Cold start completed$(NC)"

##@ Data Flow

full-data-flow: ## Start telemetry data producer
	@echo "$(BLUE)📡 Starting producer...$(NC)"
	docker compose up -d \
		producer_service
	@echo "$(GREEN)✅ Producer started - sending telemetry data$(NC)"

##@ Complete Workflows

first-run: full-datasets feature-store-setup first-training online-services cold-start full-data-flow ## Complete workflow from scratch
	@echo "$(GREEN)🎉 Complete system is running!$(NC)"
	@echo ""
	@echo "$(YELLOW)📍 All Access Points:$(NC)"
	@echo "  - Redis Insight:     http://localhost:5540"
	@echo "  - Redpanda Console:  http://localhost:8080"
	@echo "  - MLflow:            http://localhost:5000"
	@echo "  - Qdrant:            http://localhost:6333"
	@echo "  - MongoDB (compass): mongodb://admin:admin@localhost:27017/"
	@echo "  - Airflow:           http://localhost:8081 (airflow/airflow)"
	@echo "  - Feature Store:     http://localhost:8000"
	@echo "  - Inference API:     http://localhost:8001"
	@echo "  - MCP Server:        http://localhost:8020"
	@echo "  - vLLM:              http://localhost:8222"
	@echo "  - LangChain Agent:   http://localhost:8010"

online-run: online-services full-data-flow ## Start only online services (assumes setup is done)
	@echo "$(GREEN)🎉 Online system is running!$(NC)"

##@ Data Management

clean-data: ## Reset streaming state and recreate topics
	@echo "$(YELLOW)🧹 Cleaning streaming data...$(NC)"
	docker compose stop inference_service streaming_service || true
	sudo rm -rf data/offline/streaming_backfill
	sudo rm -rf data/entity_df/telemetry_data
	sudo rm -rf /tmp/quix_state/
	sudo rm -rf /inference_service/state/
	docker compose up create_offline_files
	docker exec -it redpanda_broker rpk topic delete telemetry-data predictions || true
	docker exec -it redpanda_broker rpk topic create telemetry-data -p 1 -r 1
	docker exec -it redpanda_broker rpk topic create predictions -p 1 -r 1
	docker compose start inference_service streaming_service
	@echo "$(GREEN)✅ Streaming data cleaned$(NC)"

##@ Cleanup

stop: ## Stop all services
	@echo "$(YELLOW)🛑 Stopping all services...$(NC)"
	docker compose --profile setup --profile online down
	@echo "$(GREEN)✅ All services stopped$(NC)"

stop-online: ## Stop only online services (keep infrastructure)
	@echo "$(YELLOW)🛑 Stopping online services...$(NC)"
	docker compose --profile online stop
	@echo "$(GREEN)✅ Online services stopped$(NC)"

clean: stop ## Stop services and remove volumes
	@echo "$(RED)🗑️  Removing volumes...$(NC)"
	docker compose --profile setup --profile online down -v
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-all: clean ## Complete cleanup including data directories
	@echo "$(RED)🗑️  Removing all data...$(NC)"
	sudo rm -rf redpanda_storage logs data/registry data/offline data/entity_df qdrant_data outputs
	docker volume prune -f
	@echo "$(GREEN)✅ Complete cleanup done$(NC)"

rebuild: clean-all init_linux full-datasets feature-store-setup first-training ## Complete rebuild from scratch
	@echo "$(GREEN)🔄 Rebuild completed$(NC)"

##@ Monitoring

health: ## Check health of all services
	@echo "$(BLUE)🏥 Checking service health...$(NC)"
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

logs: ## Show logs from all services
	docker compose logs -f

logs-service: ## Show logs from specific service (use: make logs-service SERVICE=<name>)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)❌ Please specify SERVICE=<service_name>$(NC)"; \
		echo "$(YELLOW)Example: make logs-service SERVICE=vllm$(NC)"; \
		exit 1; \
	fi
	docker compose logs -f $(SERVICE)

##@ Help

help: ## Display this help message
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║        Anomaly Detection Production-Ready - Makefile           ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)🚀 Quick Start (First Time):$(NC)"
	@echo "  make first-run           # Complete setup from scratch"
	@echo ""
	@echo "$(YELLOW)🔄 Quick Start (Already Setup):$(NC)"
	@echo "  make online-run          # Start services (skip setup)"
	@echo ""
	@echo "$(YELLOW)📋 Step by Step:$(NC)"
	@echo "  1. make init_linux           # Create directories"
	@echo "  2. make full-datasets        # Generate synthetic data"
	@echo "  3. make infrastructure       # Start core services"
	@echo "  4. make airflow              # Start orchestration"
	@echo "  5. make feature-store-setup  # Register features"
	@echo "  6. make first-training       # Train model"
	@echo "  7. make online-services      # Start online services"
	@echo "  8. make cold-start           # Populate Redis"
	@echo "  9. make full-data-flow       # Start producer"
	@echo ""


cd:
	sudo rm -rf data/offline/streaming_backfill \
	sudo rm -rf data/entity_df/telemetry_data \
	sudo rm -rf /tmp/quix_state/ \
	sudo rm -rf /inference_service/state/ \
	docker exec -it redpanda_broker rpk topic delete telemetry-data predictions || true \
	docker exec -it redpanda_broker rpk topic create telemetry-data -p 1 -r 1 \
	docker exec -it redpanda_broker rpk topic create predictions -p 1 -r 1 \
