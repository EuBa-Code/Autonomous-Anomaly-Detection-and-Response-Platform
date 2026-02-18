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
	
#__________________________________________________________

.PHONY: build-mlflow build-training build-all \
        mlflow-up training-up all-up \
        mlflow-down training-down all-down

# ── BUILD ──────────────────────────────────────────────
build-mlflow:
	docker-compose -f compose.yaml build mlflow

build-training:
	docker-compose -f compose.yaml build training_pipeline

# ── UP ─────────────────────────────────────────────────
mlflow-up:
	docker-compose -f compose.yaml up -d mlflow
	@echo "Waiting for MLflow to be ready..."
	@ping -n 11 127.0.0.1 > nul

training-up:
	docker-compose -f compose.yaml up -d --no-build training_pipeline

all-up: mlflow-up training-up
	@echo "MLflow e Training Pipeline avviati"

# ── DOWN ───────────────────────────────────────────────
mlflow-down:
	docker-compose -f compose.yaml down mlflow

training-down:
	docker-compose -f compose.yaml stop training_pipeline

all-down: mlflow-down training-down
	@echo "Tutti i servizi fermati"