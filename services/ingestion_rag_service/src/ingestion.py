import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from config import ingestion_settings

def load_txt_documents(data_dir: str | Path) -> List[Document]:

    if isinstance(data_dir, str):
        data_path = Path(data_dir)
    else:
        data_path = data_dir

    # VALIDATION CHECKS ON DATA FOLDER
    if not data_path.exists():
        raise FileNotFoundError(f"The folder '{data_path}' does not exist. Check settings.data_dir.")
    if not data_path.is_dir():
        raise NotADirectoryError(f"'{data_path}' is not a folder.")

    paths = sorted(p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    if not paths:
        raise FileNotFoundError(f"No .txt files found in '{data_path}'.")

    docs: List[Document] = []
    print(f"Found {len(paths)} text files.")

    for p in paths:
        try:
            loaded = TextLoader(str(p), encoding="utf-8").load()
            for d in loaded:
                d.metadata["source"] = p.name
                d.metadata["path"] = str(p.resolve())
                d.metadata["doc_type"] = "kb_txt"
            docs.extend(loaded)
        except Exception as e:
            print(f"Error loading file {p.name}: {e}", file=sys.stderr)

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def main() -> None:

    # 1. Load and Split Documents
    print("--- Starting Ingestion ---")
    raw_docs = load_txt_documents(ingestion_settings.data_dir)
    chunks = split_documents(raw_docs)

    # 2. Configure Embedding Models
    # Dense embeddings configuration using Google Generative AI model
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=ingestion_settings.embedding_model, 
        model_kwargs={'device': 'cuda'}    
    )

    # Sparse embeddings configuration using BM25 algorithm
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 3. Create Hybrid Vector Store
    print(f"Creating collection '{ingestion_settings.qdrant_collection}' and indexing...")

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=ingestion_settings.qdrant_url,
        api_key=ingestion_settings.qdrant_api_key,
        prefer_grpc=True,
        collection_name=ingestion_settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True
    )

    print("\nIngestion completed!")
    print(f"- Original documents: {len(raw_docs)}")
    print(f"- Created chunks:     {len(chunks)}")
    print(f"- Qdrant collection:  {ingestion_settings.qdrant_collection} (Hybrid)")


if __name__ == "__main__":
    main()
