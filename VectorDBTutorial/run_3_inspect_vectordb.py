# ============================================================
#  SCRIPT 03 — Inspecting the Vector Database
#
#  This script shows what is ACTUALLY inside ChromaDB:
#    - All stored IDs
#    - The metadata table (readable summary)
#    - Full vectors for two documents side-by-side
#    - A distance matrix so you can see which docs are close
# ============================================================
import math
import os
import sys
import time
import chromadb

from envsettings import getChromaDBDir

def cosine_similarity(a, b):
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(y ** 2 for y in b))
    return dot / (mag_a * mag_b)

if __name__ == "__main__":
    if os.system("cls" if os.name == "nt" else "clear") is not None:
        pass  # Clear the console for better readability

    chroma_db_dir = getChromaDBDir()
    collection_name = "EMBEDDINGS_COLLECTION"

    db = chromadb.PersistentClient(path=chroma_db_dir)
    collection = db.get_or_create_collection(name=collection_name)

    total = collection.count()
    print("=" * 60)
    print(f"VECTOR DATABASE INSPECTOR  ({total} documents stored)")
    print("=" * 60)

    # ── SECTION 1: All stored IDs ───────────────────────────────
    print("\n── All stored document IDs ──────────────────────────────")
    all_data = collection.get(include=["metadatas", "embeddings", "documents"])

    for i, (doc_id, meta) in enumerate(zip(all_data["ids"], all_data["metadatas"]), 1):
        print(f"  {i:02d}.  {doc_id:15s}  [{meta['category']:10s}]  {meta['title']}")


    # ── SECTION 2: Metadata table ───────────────────────────────
    print("\n── Metadata stored with each document ───────────────────")
    header = f"  {'ID':15s}  {'Title':30s}  {'Category':12s}  {'Source':25s}  {'Year'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        print(
            f"  {doc_id:15s}  {meta['title']:30s}  {meta['category']:12s}"
            f"  {meta['source']:25s}  {meta['year']}"
        )


    # ── SECTION 3: Show a full vector ───────────────────────────
    print("\n── Full vector for 'space_001' ───────────────────────────")
    idx   = all_data["ids"].index("space_001")
    vec   = all_data["embeddings"][idx]
    text  = all_data["documents"][idx]

    print(f"\n  Document text:\n  '{text[:100]}...'")
    dims = len(vec)
    print(f"\n  Vector length: {dims} dimensions")
    print(f"\n  Values (showing first 20 out of {dims}):")

    ROW = 5
    for i in range(0, min(20, len(vec)), ROW):
        chunk = vec[i : i + ROW]
        formatted = "  ".join(f"{v: .5f}" for v in chunk)
        print(f"  [{i:4d}-{i+ROW-1:4d}]  {formatted}")
    print(f"  ... (remaining {dims - 20} values not shown)")


    # ── SECTION 4: Two vectors side-by-side ─────────────────────
    print("\n── Two vectors side-by-side (first 10 dims) ─────────────")
    idx_a = all_data["ids"].index("space_001")
    idx_b = all_data["ids"].index("animal_001")
    vec_a = all_data["embeddings"][idx_a]
    vec_b = all_data["embeddings"][idx_b]

    print(f"\n  {'Dim':>5}  {'space_001':>12}  {'animal_001':>12}  {'Difference':>12}")
    print(f"  {'---':>5}  {'----------':>12}  {'----------':>12}  {'----------':>12}")
    for i in range(10):
        diff = vec_a[i] - vec_b[i]
        print(f"  {i:>5}  {vec_a[i]:>12.6f}  {vec_b[i]:>12.6f}  {diff:>+12.6f}")

    sim = cosine_similarity(vec_a, vec_b)
    print(f"\n  Cosine similarity between space_001 and animal_001: {sim:.4f}")
    print("  (If this is close to 1.0 → similar. Close to 0 → different.)")


    # ── SECTION 5: Mini distance matrix ─────────────────────────
    print("\n── Similarity matrix (one doc per category) ─────────────")

    sample_ids = ["space_001", "animal_001", "cooking_001", "tech_001"]
    sample_indices = [all_data["ids"].index(sid) for sid in sample_ids]
    sample_vecs    = [all_data["embeddings"][i] for i in sample_indices]
    labels         = ["Space", "Animal", "Cook", "Tech"]

    # Header row
    header_row = f"  {'':8s}" + "".join(f"  {l:8s}" for l in labels)
    print(f"\n{header_row}")

    for i, (label_i, vec_i) in enumerate(zip(labels, sample_vecs)):
        row = f"  {label_i:8s}"
        for j, vec_j in enumerate(sample_vecs):
            sim = cosine_similarity(vec_i, vec_j)
            row += f"  {sim:8.4f}"
        print(row)

    print("\n  Diagonal = 1.0 (a doc compared to itself)")
    print("  Same-topic docs score HIGHER. Cross-topic docs score LOWER.")
