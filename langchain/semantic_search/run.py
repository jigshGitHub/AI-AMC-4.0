import os
from typing import Iterable
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"

# Chunking controls how much text goes into each searchable piece.
CHUNK_SIZE = 1000 # The maximum number of characters (or tokens) each chunk can contain.
CHUNK_OVERLAP = 200 # The number of characters that consecutive chunks should share. This helps maintain context across splits.
                    # The overlap helps mitigate the possibility of separating a statement from important context related to it.

# Embedding model used to convert text into vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def ensure_data_directory(data_dir: str) -> bool:
    """
    Make sure the data directory exists.

    If it does not exist yet, create it and tell the user what to do next.
    """
    if os.path.exists(data_dir):
        return True

    os.makedirs(data_dir)
    print(f"Created '{data_dir}/'. Add PDF files there, then run ingestion again.")
    return False

def get_pdf_paths(data_dir: str) -> list[str]:
    """Return all PDF file paths inside the data directory."""
    pdf_filenames = sorted(
        filename for filename in os.listdir(data_dir) if filename.lower().endswith(".pdf")
    )
    return [os.path.join(data_dir, filename) for filename in pdf_filenames]

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
        print(f"Loading PDF: {pdf_name}")
        loader = PyPDFLoader(pdf_path) # Loads one document for each PDF page, with metadata for source and page number
        documents.extend(loader.load())

    return documents

def load_source_documents(data_dir: str = DATA_DIR) -> list[Document]:
    """
    Load all PDF documents from the project data folder.

    This function is intentionally small and readable so students can follow
    the control flow without getting lost in implementation details.
    """
    if not ensure_data_directory(data_dir):
        return []

    pdf_paths = get_pdf_paths(data_dir)
    if not pdf_paths:
        print(f"No PDF files found in '{data_dir}/'. Add some files and try again.")
        return []

    documents = load_pdf_documents(pdf_paths)
    print(f"Loaded {len(documents)} page(s) from {len(pdf_paths)} PDF file(s).")
    return documents

def get_text_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """Return a text splitter configured with the specified chunk size and overlap."""
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Return an embedding model instance based on the specified model name."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_vector_store(chunks: list[Document], embedding_model) -> Chroma:
    """Create a Chroma vector store from the given text chunks and embedding model."""
    return Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=CHROMA_DB_DIR)

def perform_ingetion(data_dir: str) :

    documents = load_source_documents(data_dir)

    # for doc in documents:
    #     print(f"- {doc.metadata['title']} (length: {len(doc.page_content)})")
    #     print(f"-----------------------------------\n{doc.page_content[:200]}...\n")

    text_splitter = get_text_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i + 1} of document '{chunk.metadata['title']}':")
    #     print(chunk)
    #     print("=" * 60)

    embedding_model = get_embedding_model(EMBEDDING_MODEL)
    print(f"Embedding model '{EMBEDDING_MODEL}' is ready to use for creating vector representations of the text chunks.")

    vector_store = create_vector_store(chunks, embedding_model)
    print(f"Vector store created successfully at '{CHROMA_DB_DIR}'.")

    print()
    print("Ingestion complete. Your documents are ready for retrieval.")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SEMANTIC SEARCH WITH LANGCHAIN")
    print("  Powered by LangChain + OpenAI")
    print("=" * 60)
    print("\nThis is a simple example of how to load a PDF document and prepare it for semantic search using LangChain.\n")
    print("The PDF document is loaded and split into smaller chunks, which can then be embedded and indexed for efficient retrieval.\n")
    print("Loaded documents:")

    # perform_ingetion(DATA_DIR)
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embedding_model(EMBEDDING_MODEL)
    )
    results = vector_store.similarity_search(
         "How were Nike's margins impacted in 2023?", k = 3
    )
    for doc in results:
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source')}\n")
