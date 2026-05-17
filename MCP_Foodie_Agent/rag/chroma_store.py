"""ChromaDB store. Seeds from cuisines.md on first run. Uses absolute paths."""
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    CUISINES_MD_PATH,
    RAG_TOP_K,
)


def _split_markdown(text: str) -> list[str]:
    chunks, current = [], []
    for line in text.split("\n"):
        if line.startswith("## "):
            if current:
                chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c]


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    ef = DefaultEmbeddingFunction()
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=ef)


def seed_if_empty() -> None:
    coll = get_collection()
    if coll.count() > 0:
        return
    text = CUISINES_MD_PATH.read_text(encoding="utf-8")
    chunks = _split_markdown(text)
    coll.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    print(f"[RAG] Seeded {len(chunks)} chunks from {CUISINES_MD_PATH}")


def retrieve(query: str, k: int = RAG_TOP_K) -> list[str]:
    coll = get_collection()
    results = coll.query(query_texts=[query], n_results=k)
    return results["documents"][0] if results["documents"] else []
