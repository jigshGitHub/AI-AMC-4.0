from __future__ import annotations
import config
import os
import applogging

from typing import Iterable
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = applogging.get_logger("real_estate_app")

def build_embedding_model() -> HuggingFaceEmbeddings:
    """Create the embedding model used by both ingestion and retrieval."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Store chunk embeddings in a local Chroma database.

    Chroma keeps:
    - the vector embeddings
    - the original chunk text
    - the metadata needed for citations
    """
    embeddings = build_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )

    logger.info(f"Saved vector database to '{config.CHROMA_DB_DIR}/'.")
    return vector_store

def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter used during chunking.

    We use RecursiveCharacterTextSplitter because it is one of the clearest and
    most common starting points for RAG projects.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split loaded pages into smaller overlapping chunks.

    Why this matters:
    - Smaller chunks make retrieval more precise
    - Overlap helps preserve context across chunk boundaries
    """
    text_splitter = build_text_splitter()
    chunks = text_splitter.split_documents(documents)

    logger.info(
        f"Created {len(chunks)} chunk(s) "
        f"(chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP})."
    )
    return chunks

def load_pdf_documents(pdf_paths: Iterable[str]) -> list[Document]:
    """
    Load every PDF page as a LangChain Document.

    Each page becomes one Document with:
    - page_content: extracted text
    - metadata: source file path and page number
    """
    documents: list[Document] = []

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
        logger.info(f"Loading PDF: {pdf_name}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    return documents

def get_pdf_paths(data_dir: str) -> list[str]:
    pdf_filenames = sorted(
        filename for filename in os.listdir(data_dir) if filename.lower().endswith(".pdf")
    )
    return [os.path.join(data_dir, filename) for filename in pdf_filenames]

def ensure_data_directory(data_dir: str) -> bool:
    if os.path.exists(data_dir):
        return True

    os.makedirs(data_dir)
    logger.info(f"Created '{data_dir}/'. Add PDF files there, then run ingestion again.")
    return False

def load_source_documents(data_dir: str = config.DATA_DIR) -> list[Document]:
    if not ensure_data_directory(data_dir):
        return []

    pdf_paths = get_pdf_paths(data_dir)
    if not pdf_paths:
        logger.error(f"No PDF files found in '{data_dir}/'. Add some files and try again.")
        return []

    documents = load_pdf_documents(pdf_paths)
    logger.info(f"Loaded {len(documents)} page(s) from {len(pdf_paths)} PDF file(s).")
    return documents

def run_ingestion() -> Chroma | None:
    documents = load_source_documents(config.DATA_DIR)
    if not documents:
        return None
    logger.success("PDF documents loaded successfully")

    chunks = split_documents(documents)
    logger.success("Split documents DONE")
    
    vector_store = create_vector_store(chunks)
    logger.success("Ingestion complete. Documents are ready for retrieval.")
    return vector_store


if __name__ == "__main__":
    run_ingestion()