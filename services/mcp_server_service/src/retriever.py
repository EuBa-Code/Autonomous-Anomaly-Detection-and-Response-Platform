"""
retriever.py
-------------
Implements the full retrieval pipeline from the architecture:

  Query embedding
      ├── Dense search  (Qdrant HNSW)  ─┐
      └── Sparse search (Qdrant BM42) ──┤ Hybrid search
                                         │
                                    RRF Fusion
                                         │
                                    Re-ranking  (cross-encoder)
                                         │
                                    Contexts  → API
"""

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedSparseVector,
    NamedVector,
)
from sentence_transformers import CrossEncoder

from config import get_config
from .embeddings import embed_query

logger = logging.getLogger(__name__)


# ── Qdrant client ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_qdrant() -> QdrantClient:
    cfg = get_config()
    qcfg = cfg["qdrant"]
    kwargs = {"url": qcfg["url"]}
    if api_key := qcfg.get("api_key"):
        kwargs["api_key"] = api_key
    client = QdrantClient(**kwargs)
    logger.info("Qdrant client connected to %s", qcfg["url"])
    return client


# ── Cross-encoder reranker ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder | None:
    cfg = get_config()
    rcfg = cfg["reranker"]
    if not rcfg.get("enabled", True):
        return None
    model_name = rcfg["model"]
    device = rcfg.get("device", "cpu")
    logger.info("Loading cross-encoder reranker: %s on %s", model_name, device)
    return CrossEncoder(model_name, device=device, max_length=rcfg.get("max_length", 512))


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fuse(
    dense_hits: list,
    sparse_hits: list,
    k: int = 60,
) -> list[tuple[str, float, str]]:
    """
    Merge dense and sparse result lists using Reciprocal Rank Fusion.
    Returns list of (id, rrf_score, payload_text) sorted desc by score.
    """
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}

    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        texts[pid] = hit.payload.get("text", "")

    for rank, hit in enumerate(sparse_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        if pid not in texts:
            texts[pid] = hit.payload.get("text", "")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(pid, score, texts[pid]) for pid, score in ranked]


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> tuple[list[str], list[float]]:
    """
    Full hybrid retrieval pipeline:
      1. Embed query (dense + sparse)
      2. Search Qdrant (dense + sparse separately)
      3. Fuse with RRF
      4. Rerank with cross-encoder
      5. Return top_k (texts, scores)
    """
    cfg = get_config()
    qcfg = cfg["qdrant"]
    rcfg = cfg["retrieval"]

    collection = qcfg["collection_name"]
    dense_vec_name = qcfg["dense_vector_name"]
    sparse_vec_name = qcfg["sparse_vector_name"]
    top_k_dense = rcfg["top_k_dense"]
    top_k_sparse = rcfg["top_k_sparse"]

    # ── Step 1: Embed ──────────────────────────────────────────────────────────
    logger.info("Embedding query: '%s'", query[:80])
    emb = embed_query(query)

    client = _get_qdrant()

    # ── Step 2a: Dense search ──────────────────────────────────────────────────
    dense_hits = client.search(
        collection_name=collection,
        query_vector=NamedVector(name=dense_vec_name, vector=emb.dense),
        limit=top_k_dense,
        with_payload=True,
    )
    logger.debug("Dense search returned %d hits", len(dense_hits))

    # ── Step 2b: Sparse search ─────────────────────────────────────────────────
    sparse_hits = client.search(
        collection_name=collection,
        query_vector=NamedSparseVector(name=sparse_vec_name, vector=emb.sparse),
        limit=top_k_sparse,
        with_payload=True,
    )
    logger.debug("Sparse search returned %d hits", len(sparse_hits))

    # ── Step 3: RRF fusion ─────────────────────────────────────────────────────
    k_rrf = rcfg.get("rrf_k", 60)
    fused = _rrf_fuse(dense_hits, sparse_hits, k=k_rrf)
    logger.info("After RRF fusion: %d unique candidates", len(fused))

    if not fused:
        return [], []

    # ── Step 4: Cross-encoder reranking ───────────────────────────────────────
    reranker = _get_reranker()

    if reranker is not None:
        candidate_texts = [text for _, _, text in fused]
        pairs = [[query, text] for text in candidate_texts]
        ce_scores: list[float] = reranker.predict(pairs).tolist()

        reranked = sorted(
            zip(candidate_texts, ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        texts = [t for t, _ in reranked[:top_k]]
        scores = [s for _, s in reranked[:top_k]]
        logger.info("Reranked to top %d results", len(texts))
    else:
        # No reranker — just return RRF top-k
        texts = [text for _, _, text in fused[:top_k]]
        scores = [score for _, score, _ in fused[:top_k]]

    return texts, scores


def health_check() -> bool:
    """Return True if Qdrant is reachable and the collection exists."""
    try:
        cfg = get_config()
        client = _get_qdrant()
        collections = [c.name for c in client.get_collections().collections]
        return cfg["qdrant"]["collection_name"] in collections
    except Exception as exc:
        logger.error("Qdrant health check failed: %s", exc)
        return False