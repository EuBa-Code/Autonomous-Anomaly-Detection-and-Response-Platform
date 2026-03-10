from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from qdrant.config import retrieval_settings
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."

    return "\n\n".join(
        f"(Source: {d.metadata.get('source', 'unknown')})\n{d.page_content.strip()}"
        for d in docs
    )

dense_embeddings_model = HuggingFaceEmbeddings(
        model=retrieval_settings.embedding_model
    )

sparse_embeddings_model = FastEmbedSparse(model_name="Qdrant/bm25")

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

vector_store = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings_model,  # dense embeddings
        sparse_embedding=sparse_embeddings_model,  # sparse embeddings
        url=retrieval_settings.qdrant_url,
        api_key=retrieval_settings.qdrant_api_key,
        prefer_grpc=False,
        collection_name=retrieval_settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID  # set hybrid search mode
)

base_retriever = vector_store.as_retriever(  # retriever for step 1 (before reranking)
        search_type="mmr",  # search technique: MMR (Maximal Marginal Relevance)
        search_kwargs={
            "k": 6,  # reselect top 6 based on MMR
            "fetch_k": 20,  # documents extracted in first step (hybrid search)
        }
)

ranker_client = Ranker(  # reranking model
        model_name="ms-marco-TinyBERT-L-2-v2",  # model name
        cache_dir="models/flashrank"  # (Optional) Save model here
)

compressor = FlashrankRerank(
        client=ranker_client,  # reranker model
        top_n=4  # top k most relevant documents in output
)

compression_retriever = ContextualCompressionRetriever(
        # implements multi-step retrieval pipeline
        base_compressor=compressor,  # reranking step
        base_retriever=base_retriever  # base retrieval step (use previously defined retriever)
)
