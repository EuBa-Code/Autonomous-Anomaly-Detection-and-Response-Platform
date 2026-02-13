create_datasets:
	uv run -m data.prepare_data --group dataset-creation-2
	
hist_service:
	docker compose up hist_ingestion

hist_service_down:
	docker compose down hist_ingestion

run_all:
	uv sync
	docker compose up --build create_datasets hist_ingestion


create_datasets_2:
	docker compose up --build create_datasets