# ==============================================================================
# WASHING MACHINES ANOMALY DETECTION
# ==============================================================================
.PHONY: help start setup airflow online stop clean \
        create_offline create_datasets data_engineering \
        infra-up feast-apply batch-pipeline training \
        logs-airflow logs-online logs-all

SEP := ----------------------------------------------------------------

# ==============================================================================
# HELP
# ==============================================================================
help:
	@echo ""
	@echo "  WASHING MACHINES ANOMALY DETECTION"
	@echo "  $(SEP)"
	@echo "  make start            Full pipeline from scratch (single command)"
	@echo ""
	@echo "  Individual phases (run in order if needed separately):"
	@echo "  make setup            Offline: data -> features -> train"
	@echo "  make airflow          Start Airflow (handles cold-start materialization)"
	@echo "  make online           Start inference, streaming, producer"
	@echo ""
	@echo "  Granular steps inside setup:"
	@echo "  make create_offline   Generate parquet files for offline store"
	@echo "  make create_datasets  Synthesize raw machine telemetry data"
	@echo "  make data_engineering Run PySpark feature engineering"
	@echo "  make infra-up         Bring up Redis, Redpanda, MLflow, Qdrant"
	@echo "  make feast-apply      Register Feast feature definitions + ingest RAG docs"
	@echo "  make batch-pipeline   Run Spark batch feature pipeline -> offline store"
	@echo "  make training         Fit preprocessor + train Isolation Forest"
	@echo ""
	@echo "  Lifecycle:"
	@echo "  make stop             Gracefully stop all running containers"
	@echo "  make clean            Stop everything + wipe all volumes (destructive)"
	@echo ""
	@echo "  Logs:"
	@echo "  make logs-airflow     Follow Airflow webserver + scheduler logs"
	@echo "  make logs-online      Follow online services logs"
	@echo "  make logs-all         Follow every container log"
	@echo "  $(SEP)"
	@echo ""

# ==============================================================================
# SINGLE ENTRY POINT
#   1. Offline setup  (data gen -> feature engineering -> feast apply -> train)
#   2. Airflow        (init DB -> webserver + scheduler)
#                     The DAG cold-start branch materializes offline->Redis
#                     on its very first run, no extra step needed.
#   3. Online         (feature server -> streaming -> inference -> producer)
# ==============================================================================
start: setup airflow online
	@echo ""
	@echo "  [OK] Full pipeline is up and running"
	@echo "  $(SEP)"
	@echo "  Airflow UI    -> http://localhost:8081   (airflow / airflow)"
	@echo "  MLflow UI     -> http://localhost:5000"
	@echo "  Inference API -> http://localhost:8001"
	@echo "  Feast Server  -> http://localhost:8000"
	@echo "  Redpanda UI   -> http://localhost:8080"
	@echo "  Redis Insight -> http://localhost:5540"
	@echo "  $(SEP)"
	@echo ""

# ==============================================================================
# PHASE 1 -- OFFLINE SETUP
# Generates synthetic data, engineers features, applies Feast schema,
# writes to the offline store, and trains the model.
# Feast materialization (offline -> Redis) is intentionally left to Airflow's
# cold-start DAG branch so all scheduling is owned in one place.
# ==============================================================================
setup: create_offline create_datasets data_engineering infra-up feast-apply batch-pipeline training
	@echo "  [OK] Offline setup complete"

create_offline:
	@echo "  [1/7] Creating offline parquet files..."
	uv run --group offline-files -m utils.create_offline_files

create_datasets:
	@echo "  [2/7] Generating synthetic machine telemetry datasets..."
	docker compose up --build create_datasets

data_engineering:
	@echo "  [3/7] Running PySpark data engineering pipeline..."
	docker compose up --build data_engineering

infra-up:
	@echo "  [4/7] Starting infrastructure (Redis, Redpanda, MLflow, Qdrant)..."
	docker compose up -d --build redis redpanda redpanda-console redpanda-init redis-insight
	@echo "  Waiting 15s for services to become healthy..."
	@sleep 15

feast-apply:
	@echo "  [5/7] Registering Feast feature definitions + ingesting RAG docs..."
	docker compose --profile setup up --build --abort-on-container-exit feature_store_apply 

batch-pipeline:
	@echo "  [6/7] Running Spark batch feature pipeline -> offline store..."
	docker compose up --build --abort-on-container-exit batch_feature_pipeline

training:
	@echo "  [7/7] Training Isolation Forest model via MLflow..."
	docker compose up --build --abort-on-container-exit training_service

# ==============================================================================
# PHASE 2 -- AIRFLOW
# Starts Postgres, initialises the DB, creates the admin user, then brings up
# the webserver and scheduler.
#
# Cold-start note: on the very first DAG trigger the check_if_first_time
# branch routes to setup_task, which calls store.materialize_incremental()
# to push all offline-store rows into Redis. The Variable
# batch_pipeline_initialized is then flipped to "true" so every subsequent
# daily run goes straight to the Spark DockerOperator instead.
# ==============================================================================
airflow:
	@echo "  [Airflow] Starting Postgres and initialising database..."
	docker compose up -d --build postgres_airflow
	@echo "  Waiting for Postgres to be healthy..."
	@sleep 10
	@echo "  [Airflow] Creating admin user..."
	docker compose up --build --abort-on-container-exit airflow-init
	@echo "  [Airflow] Starting webserver and scheduler..."
	docker compose up -d --build airflow-webserver airflow-scheduler
	@echo "  [OK] Airflow is up -> http://localhost:8081  (airflow / airflow)"

# ==============================================================================
# PHASE 3 -- ONLINE SERVICES
# Feature server (Feast HTTP), Quix streaming pipeline, FastAPI inference,
# and the telemetry producer.
# ==============================================================================
online:
	@echo "  [Online] Starting feature server, streaming service, inference API..."
	docker compose --profile online up -d --build
	@echo "  [Online] Starting telemetry producer..."
	docker compose up -d --build producer_service
	@echo "  [OK] Online services are up"

# ==============================================================================
# LIFECYCLE
# ==============================================================================
stop:
	@echo "  Stopping all services..."
	docker compose --profile online down
	docker compose down
	@echo "  [OK] All containers stopped"

clean:
	@echo "  Deep clean: removing containers, networks, and ALL volumes..."
	docker compose --profile online down -v
	docker compose down -v
	@echo "  [OK] Environment fully reset"

# ==============================================================================
# LOGS
# ==============================================================================
logs-airflow:
	docker compose logs -f airflow-webserver airflow-scheduler

logs-online:
	docker compose --profile online logs -f

logs-all:
	docker compose --profile online logs -f &
	docker compose logs -f

# ------------ DEBUG ACTIVATE DAG WITH AIRFLOW GUI

debug_1:
	docker compose up --build --wait redis redis-insight redpanda redpanda-console feature_store_service

debug_2:
	docker compose run --rm cold_start

debug_3:
	docker compose up --build -d streaming_service

debug_4:
	docker compose up --build producer_service -d 

debug_5:
	docker compose up --build retraining_service


# ----------- INFERENCE 


.PHONY: test-inference down-inference

# Starts only what is needed to test the inference pipeline end-to-end:
#   redis             → online feature store
#   redpanda          → message broker
#   redpanda-init     → creates telemetry-data and predictions topics
#   mlflow            → model registry (inference loads the model at startup)
#   feature_store_service → Feast HTTP server (inference fetches features from here)
#   streaming_service → computes rolling window features and pushes to Redis
#   inference_service → consumes telemetry-data, scores, publishes to predictions
#   producer_service  → sends test telemetry messages
#   redpanda-console  → UI to inspect telemetry-data and predictions topics
test-inference:
	docker compose --profile online up --build \
		redis \
		redpanda \
		redpanda-console \
		redpanda-init \
		mlflow \
		feature_store_service \
		inference_service \
		streaming_service \
		create_offline_files \
		cold_start \
		producer_service
	@echo ""
	@echo "Redpanda console : http://localhost:8080"
	@echo "Feast server     : http://localhost:8000/health"
	@echo "MLflow UI        : http://localhost:5000"

down-inference:
	docker compose --profile online down \
		redis redis-insight redpanda redpanda-console redpanda-init mlflow \
		feature_store_service streaming_service inference_service producer_service


ingestion_rag:
	docker compose up -d --build qdrant ingestion_rag

block_2:
	docker compose up --build redpanda redpanda-console qdrant mongodb if_anomaly vllm mcp_server langchain_service \

# ---------------------------- FINAL DEBUG

init:
	sudo chown -R 101:101 ./redpanda_storage && mkdir -p .logs && chmod 777 .logs/

full_datasets:
	docker compose up --build \
		create_datasets \
		data_engineering \
		create_offline_files

first_training:
	docker compose up --build --abort-on-container-exit training_service 

full_architecture:
	@echo "  🚀 Full Architecture (WITH Airflow)"
	docker compose up \
		redis \
		redpanda \
		redpanda-console \
		redpanda-init \
		mlflow \
		qdrant \
		mongodb \
		postgres_airflow \
		airflow-init \
		airflow-webserver \
		airflow-scheduler \
		feature_store_apply \
		feature_store_service \
		streaming_service \
		inference_service \
		ingestion_rag \
		mcp_server \
		vllm \
		langchain_service \
		if_anomaly 

cold_start:
	docker compose up --build cold_start

full_data_flow:
	docker compose up \
		producer_service


# sudo chown -R 101:101 ./redpanda_storage
# 