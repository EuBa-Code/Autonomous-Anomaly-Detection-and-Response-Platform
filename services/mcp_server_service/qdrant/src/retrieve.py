from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import retrieval_settings
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "Nessun documento rilevante trovato."

    return "\n\n".join(
        f"(Fonte: {d.metadata.get('source', 'unknown')})\n{d.page_content.strip()}"
        for d in docs
    )

dense_embeddings_model = HuggingFaceEmbeddings(
        model=retrieval_settings.embedding_model
    )