from pathlib import Path

from langchain_core.documents import Document


def test_run_ingestion_creates_vector_store(monkeypatch, tmp_path):
    """Smoke test for ingestion.run_ingestion.

    This test avoids real embedding/vector DB work by monkeypatching the
    heavy operations (PDF loading, splitting, and vector store creation).
    It verifies that `run_ingestion` returns the object produced by
    `create_vector_store` when PDF files exist in `DATA_DIR`.
    """
    from RAG.RealEstate import ingestion

    # Create a dummy PDF file (content not used because we mock loader)
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")

    # Point the ingestion module to our tmp data dir
    monkeypatch.setattr(ingestion.config, "DATA_DIR", str(tmp_path))

    # Mock load_pdf_documents to return one Document per page
    def fake_load_pdf_documents(paths):
        return [Document(page_content="page text", metadata={"source": str(pdf_path), "page": 0})]

    monkeypatch.setattr(ingestion, "load_pdf_documents", fake_load_pdf_documents)

    # Mock split_documents to return a single chunk document
    def fake_split_documents(docs):
        return [Document(page_content="chunk text", metadata={"source": "dummy.pdf", "page": 0})]

    monkeypatch.setattr(ingestion, "split_documents", fake_split_documents)

    # Mock create_vector_store to avoid creating a real Chroma DB
    sentinel = object()
    monkeypatch.setattr(ingestion, "create_vector_store", lambda chunks: sentinel)

    result = ingestion.run_ingestion()
    assert result is sentinel
