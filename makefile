create_datasets:
	uv run -m data.prepare_data --group dataset-creation
	
hist_service:
	docker compose up hist_ingestion

hist_service_down:
	docker compose down hist_ingestion



run_all_hi:
	uv run --group dataset-creation -m data.prepare_data
	docker compose up hist_ingestion

create_datasets_2:
	docker compose up --build create_datasets
	

.PHONY: mlflow-up training-up all-up mlflow-down training-down all-down logs-mlflow logs-training

# MLflow commands
mlflow-up:
	docker-compose -f compose.yaml up -d mlflow
	@echo "Waiting for MLflow to be ready..."
	@sleep 10

mlflow-down:
	docker-compose -f compose.yaml down

# Training Pipeline commands
training-up:
	docker-compose -f compose.yaml up -d training_pipeline

training-down:
	docker-compose -f compose.yaml down

# Combined commands
all-up: mlflow-up training-up
	@echo "✅ MLflow and Training Pipeline are running"

all-down:
	docker-compose -f compose.yaml down -v
	@echo "✅ All services stopped and volumes removed"

# Logs commands
logs-mlflow:
	docker-compose -f compose.yaml logs -f mlflow

logs-training:
	docker-compose -f compose.yaml logs -f training_pipeline

logs-all:
	docker-compose -f compose.yaml logs -f