"""
embeddings.py
--------------
Provides two embedding backends that match the architecture diagram:

  ┌─────────────────────────┐
  │   Query embedding       │
  │  ┌────────┐ ┌─────────┐ │
  │  │ Dense  │ │ Sparse  │ │
  │  │(Google)│ │ (BM42)  │ │
  │  └────────┘ └─────────┘ │
  └─────────────────────────┘

Dense  → Google text-embedding-004  (768-dim)
Sparse → Qdrant/bm42-all-minilm-l6-v2-attentions via FastEmbed
"""

import logging
import os
from functools import lru_cache
from typing import NamedTuple

import google.generativeai as genai
from fastembed.sparse import SparseTextEmbedding
from qdrant_client.models import SparseVector

from .config import get_config

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

class EmbeddingResult(NamedTuple):
    dense: list[float]          # 768-dimensional Google embedding
    sparse: SparseVector        # Qdrant SparseVector (indices + values)


# ── Dense (Google) ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_google_model() -> None:
    """Configure the Google Generative AI client once."""
    cfg = get_config()
    api_key = cfg["embeddings"]["dense"].get("google_api_key") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for dense embeddings")
    genai.configure(api_key=api_key)
    logger.info("Google Generative AI client configured")


def embed_dense(text: str, is_query: bool = True) -> list[float]:
    """Return a 768-dim dense embedding from Google text-embedding-004."""
    _get_google_model()
    cfg = get_config()
    dcfg = cfg["embeddings"]["dense"]

    task_type = dcfg["query_task_type"] if is_query else dcfg["task_type"]
    result = genai.embed_content(
        model=dcfg["model"],
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def embed_dense_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Batch dense embeddings (documents at ingestion time)."""
    return [embed_dense(t, is_query=is_query) for t in texts]


# ── Sparse (BM42 via FastEmbed) ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_sparse_model() -> SparseTextEmbedding:
    cfg = get_config()
    model_name = cfg["embeddings"]["sparse"]["model"]
    logger.info("Loading sparse embedding model: %s", model_name)
    return SparseTextEmbedding(model_name=model_name)


def embed_sparse(text: str) -> SparseVector:
    """Return a Qdrant SparseVector using BM42."""
    model = _get_sparse_model()
    # fastembed returns a generator; consume first result
    result = next(model.embed([text]))
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def embed_sparse_batch(texts: list[str]) -> list[SparseVector]:
    """Batch sparse embeddings."""
    model = _get_sparse_model()
    return [
        SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
        for r in model.embed(texts)
    ]


# ── Combined ──────────────────────────────────────────────────────────────────

def embed_query(text: str) -> EmbeddingResult:
    """Produce both dense + sparse embeddings for a query string."""
    dense = embed_dense(text, is_query=True)
    sparse = embed_sparse(text)
    logger.debug("Embedded query (%d chars) → dense=%d dims, sparse=%d nnz",
                 len(text), len(dense), len(sparse.indices))
    return EmbeddingResult(dense=dense, sparse=sparse)