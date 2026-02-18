help:
	@echo "=============================================="
	@echo "DATA PIPELINE & FEATURE STORE MAKEFILE COMMANDS"
	@echo "=============================================="
	@echo ""
	@echo "1) create_datasets_2"
	@echo "   Sync Python environment and run dataset creation pipeline (group 2)."
	@echo "   Command:"
	@echo "     make create_datasets_2"
	@echo ""
	@echo "2) hist_ingestion"
	@echo "   Build and run historical data ingestion container."
	@echo "   Command:"
	@echo "     make hist_ingestion"
	@echo ""
	@echo "3) create_datasets"
	@echo "   Build and run dataset creation container."
	@echo "   Command:"
	@echo "     make create_datasets"
	@echo ""
	@echo "4) run_all"
	@echo "   Build and run dataset creation and historical ingestion together."
	@echo "   Command:"
	@echo "     make run_all"
	@echo ""
	@echo "5) run_feature_store"
	@echo "   Start Redis and Redpanda, apply feature store schema, and run the service."
	@echo "   Then prints service logs."
	@echo "   Command:"
	@echo "     make run_feature_store"
	@echo ""
	@echo "6) test_feature_store"
	@echo "   Run feature store integration tests in an ephemeral container."
	@echo "   Command:"
	@echo "     make test_feature_store"
	@echo ""
	@echo "=============================================="



create_datasets_2:
	uv sync
	uv run -m data.prepare_data --group dataset-creation-2
	
create_datasets:
	uv sync
	docker compose up --build create_datasets
	docker compose up --build data_engineering

data_engineering:
	uv sync
	docker compose up --build data_engineering

run_all:
	uv sync
	docker compose up --build create_datasets hist_ingestion

run_feature_store:
	uv sync
	uv run -m --group data-offline utils.create_offline_files 
	docker compose up -d redis redpanda
	docker compose up --build --abort-on-container-exit feature_store_apply
	docker compose up --build -d feature_store_service
	docker compose logs feature_store_service

test_feature_store: 
	docker compose run --rm feature_store_service uv run -m src.test_functionality

