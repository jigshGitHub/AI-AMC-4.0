# ============================================================
#  SCRIPT 05 — Querying the Vector Database
#
#  This script shows exactly how retrieval works:
#    1. Your question is embedded into a vector
#    2. ChromaDB compares it to every stored vector
#    3. The closest matches (by cosine distance) are returned
#    4. You can see the distance scores and metadata
# ============================================================

import os
import sys
import chromadb
from envsettings import getEmbeddingModel, getOpenAIClient, getChromaDBDir

# ── Demo queries ─────────────────────────────────────────────
DEMO_QUERIES = [
    "How do rockets travel to other planets?",
    "What makes bees so important for nature?",
    "How does bread rise when baking?",
    "What is a neural network?",
    "Which animals live in the coldest places on Earth?",
]

TOP_K = 3   # How many results to return for each query
def embed_text(text: str) -> list[float]:
    client = getOpenAIClient()
    response = client.embeddings.create(
        input=text,
        model=getEmbeddingModel()
    )
    return response.data[0].embedding

def run_query(question: str) -> None:
    print("\n" + "=" * 60)
    print(f"QUERY: {question}")
    print("=" * 60)

    # Step 1 — embed the query
    print("\nStep 1 — Embedding the query...")
    q_vector = embed_text(question)
    print(f"  Query vector: [{q_vector[0]:.5f}, {q_vector[1]:.5f}, "
          f"{q_vector[2]:.5f}, ... ] ({len(q_vector)} dims)")

    # Step 2 — search ChromaDB
    print(f"\nStep 2 — Searching for top {TOP_K} nearest neighbours...")
    results = collection.query(
        query_embeddings = [q_vector],
        n_results        = TOP_K,
        include          = ["documents", "metadatas", "distances"],
    )

    doc_ids   = results["ids"][0]
    distances = results["distances"][0]    # lower = more similar
    metas     = results["metadatas"][0]
    docs      = results["documents"][0]

    # ChromaDB returns L2 distance by default.
    # Convert to a 0-1 similarity score for readability.
    # similarity ≈ 1 / (1 + distance) for an intuitive display
    similarities = [1 / (1 + d) for d in distances]

    # Step 3 — display results
    print(f"\nStep 3 — Results ranked by similarity:\n")

    for rank, (doc_id, dist, sim, meta, text) in enumerate(
        zip(doc_ids, distances, similarities, metas, docs), start=1
    ):
        bar = "█" * int(sim * 30)
        print(f"  Rank #{rank}  ──────────────────────────────────────")
        print(f"    ID          : {doc_id}")
        print(f"    Title       : {meta['title']}")
        print(f"    Category    : {meta['category']}")
        print(f"    Source      : {meta['source']}  ({meta['year']})")
        print(f"    Distance    : {dist:.6f}  (lower = closer)")
        print(f"    Similarity  : {sim:.4f}  {bar}")
        print(f"    Text snippet: {text[:110]}...")
        print()

    print("  → ChromaDB found these by measuring the distance between")
    print("    the query vector and every stored document vector.")
    print("    No keyword matching — pure geometric proximity!")

if __name__ == "__main__":
    if os.system("cls" if os.name == "nt" else "clear") is not None:
        pass  # Clear the console for better readability

    chroma_db_dir = getChromaDBDir()
    collection_name = "EMBEDDINGS_COLLECTION"

    db = chromadb.PersistentClient(path=chroma_db_dir)
    collection = db.get_or_create_collection(name=collection_name)

    # ── Run all demo queries ─────────────────────────────────────
    print("=" * 60)
    print("VECTOR DB QUERY DEMO")
    print("=" * 60)
    print(f"\nDatabase has {collection.count()} documents.")
    print(f"Retrieving top {TOP_K} results for each query.\n")

    for q in DEMO_QUERIES:
        run_query(q)

    # ── Interactive mode ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRY YOUR OWN QUERY")
    print("=" * 60)
    print("(Type 'quit' to exit)\n")

    while True:
        user_q = input("Your question: ").strip()
        if user_q.lower() in ("quit", "exit", "q", ""):
            break
        run_query(user_q)
