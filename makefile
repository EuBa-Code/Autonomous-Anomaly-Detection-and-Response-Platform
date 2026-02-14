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
	docker compose up --build create_datasets hist_ingestion

run_feature_store:
	uv sync	
	docker-compose up -d redis redpanda
	docker ps

	docker-compose up --build feature_store_apply
	docker logs feature_store_apply

	docker-compose up --build -d feature_store_service
	docker logs -f feature_store_service	

	curl http://localhost:8001/health
