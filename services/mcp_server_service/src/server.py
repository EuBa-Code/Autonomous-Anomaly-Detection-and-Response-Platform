"""
MCP Server

Consuma predictions da Redpanda (solo is_anomaly == -1),
fa hybrid search su Qdrant, re-ranking con FlashRank,
e chiama un LLM via OpenRouter per generare una risposta.
"""
import os
import json
import logging
import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from confluent_kafka import Consumer, KafkaException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from flashrank import Ranker, RerankRequest
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCPServer")


class Settings(BaseSettings):
    kafka_bootstrap_servers: str = "redpanda:9092"
    topic_predictions: str = "predictions"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "knowledge_base"
    qdrant_api_key: str | None = None
    google_api_key: str
    gemini_embeddings_model: str = "models/gemini-embedding-001"
    openrouter_api_key: str
    openrouter_model: str = "mistralai/mistral-7b-instruct:free"

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(title="MCP Server")


# ============================================================================
# RAG PIPELINE
# ============================================================================

def get_vector_store() -> QdrantVectorStore:
    dense = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embeddings_model,
        google_api_key=settings.google_api_key,
    )
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")

    return QdrantVectorStore.from_existing_collection(
        embedding=dense,
        sparse_embedding=sparse,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID,
    )


def retrieve_context(query: str, top_k: int = 10) -> str:
    """Hybrid search su Qdrant + reranking con FlashRank."""
    vector_store = get_vector_store()

    # Hybrid search
    docs = vector_store.similarity_search(query, k=top_k)

    # Reranking
    ranker = Ranker()
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)

    # Prendi i top 3 dopo il reranking
    top_docs = sorted(reranked, key=lambda x: x["score"], reverse=True)[:3]
    context = "\n\n".join([p["text"] for p in top_docs])

    return context


def call_llm(machine_id: str, anomaly_score: float, context: str) -> str:
    """Chiama il LLM via OpenRouter con il contesto recuperato."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    prompt = f"""Sei un esperto di manutenzione industriale. 
Una macchina ha rilevato un'anomalia.

Macchina ID: {machine_id}
Anomaly Score: {anomaly_score}

Contesto dalla knowledge base:
{context}

Analizza l'anomalia e suggerisci le azioni correttive da intraprendere."""

    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def handle_anomaly(prediction: dict):
    """Pipeline completa: retrieve context + LLM."""
    machine_id = prediction.get("machine_id")
    anomaly_score = prediction.get("anomaly_score")

    logger.info(f"Handling anomaly for machine {machine_id}")

    query = f"anomalia macchina {machine_id} score {anomaly_score}"
    context = retrieve_context(query)
    answer = call_llm(machine_id, anomaly_score, context)

    logger.info(f"LLM Answer for machine {machine_id}:\n{answer}")


# ============================================================================
# KAFKA CONSUMER (gira in thread separato)
# ============================================================================

def consume_predictions():
    consumer = Consumer({
        'bootstrap.servers': settings.kafka_bootstrap_servers,
        'group.id': 'mcp-server-group',
        'auto.offset.reset': 'earliest',
    })
    consumer.subscribe([settings.topic_predictions])
    logger.info("Subscribed to topic 'predictions', waiting for anomalies...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            prediction = json.loads(msg.value().decode("utf-8"))

            if prediction.get("is_anomaly") == -1:
                logger.info(f"Anomaly received: {prediction}")
                handle_anomaly(prediction)

    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        consumer.close()


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=consume_predictions, daemon=True)
    thread.start()
    logger.info("Kafka consumer thread started")


# ============================================================================
# API
# ============================================================================

class AnomalyRequest(BaseModel):
    machine_id: str
    is_anomaly: int
    anomaly_score: float


@app.post("/anomaly")
def anomaly_endpoint(request: AnomalyRequest):
    """Endpoint manuale per triggerare la pipeline (utile per testing)."""
    handle_anomaly(request.dict())
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)