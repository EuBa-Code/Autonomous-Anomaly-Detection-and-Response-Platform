create_datasets_2:
	uv sync
	uv run -m data.prepare_data --group dataset-creation-2
	
hist_ingestion:
	uv sync
	docker compose up --build hist_ingestion

create_datasets:
	uv sync
	docker compose up --build create_datasets

run_all:
	uv sync
	docker compose up --build create_datasets hist_ingestion feature_store_apply

streaming:
	uv sync
	docker compose up --build streaming_service

load_features:
	uv sync
	docker compose up --build feature_loader

# NEW: Complete reset and rebuild of the environment
reboot:
	docker compose down -v --remove-orphans
	docker compose up --build -d

# NEW: Deep cleaning of docker system (use with caution)
clean_all:
	docker compose down -v --remove-orphans
	docker system prune -f --volumes
