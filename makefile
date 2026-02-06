create_datasets:
	uv run -m data.prepare_data --group dataset-creation
	
hist_service:
	docker compose up hist_ingestion

hist_service_down:
	docker compose down hist_ingestion

