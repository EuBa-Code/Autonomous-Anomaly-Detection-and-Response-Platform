create_datasets:
	uv run -m data.prepare_data --group dataset-creation
	
hist_service:
	docker compose up hist_ingestion

hist_service_down:
	docker compose down hist_ingestion

redis_up:
	docker compose up redis -d

redis_down:
	docker compose stop redis

run_all_hi:
	uv run --group dataset-creation -m data.prepare_data
	docker compose up hist_ingestion

# NEW: Complete reset and rebuild of the environment
reboot:
	docker compose down -v --remove-orphans
	docker compose up --build -d

# NEW: Deep cleaning of docker system (use with caution)
clean_all:
	docker compose down -v --remove-orphans
	docker system prune -f --volumes