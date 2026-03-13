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

dense_embeddings_model = HuggingFaceEmbeddings(model_name=retrieval_settings.embedding_model)

sparse_embeddings_model = FastEmbedSparse(model_name="Qdrant/bm25")

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")


def get_retriever():
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings_model,
        sparse_embedding=sparse_embeddings_model,
        url=retrieval_settings.qdrant_url,
        api_key=retrieval_settings.qdrant_api_key,
        prefer_grpc=False,
        collection_name=retrieval_settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID
    )
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )
    compressor = FlashrankRerank(
        client=Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="models/flashrank"),
        top_n=4
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

compression_retriever = get_retriever()  # still runs at import, but now easier to wrap in try/except