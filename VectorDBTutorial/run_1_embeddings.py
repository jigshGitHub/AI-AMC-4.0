import os
import sys
import math
from envsettings import getEmbeddingModel, getOpenAIClient

def embed_text(text):
    client = getOpenAIClient()
    response = client.embeddings.create(
        input=text,
        model=getEmbeddingModel()
    )
    return response.data[0].embedding

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    return dot / (mag_a * mag_b)

if __name__ == "__main__":
    if os.system("cls" if os.name == "nt" else "clear") is not None:
        pass  # Clear the console for better readability
    print("=" * 60)
    print("STEP 1 — Embedding a single sentence")
    print("=" * 60)
    sentence = "The cat sat on the mat."
    vector = embed_text(sentence)
    print(f"\nSentence : '{sentence}'")
    print(f"Dimensions: {len(vector)}")          # 1536 dims with ada-002
    print(f"\nFirst 10 values of the vector:")
    for i, val in enumerate(vector[:10]):
        print(f"  [{i:4d}]  {val: .6f}")
    print(f"  ...  (and {len(vector) - 10} more numbers)")

    # ── Step 2: Compare similar vs different sentences ──────────
    print("\n" + "=" * 60)
    print("STEP 2 — Similarity: do similar sentences have similar vectors?")
    print("=" * 60)

    sentences = {
        "A": "I love eating pizza.",
        "B": "Pizza is my favourite food.",          # similar to A
        "C": "Rockets are launched into outer space.",  # very different
    }

    # Compute embeddings for all sentences
    print("\nEmbedding three sentences — please wait...")
    vectors = {key: embed_text(value) for key, value in sentences.items()}

    pairs = [("A", "B"), ("A", "C"), ("B", "C")]

    print()
    for x, y in pairs:
        sim = cosine_similarity(vectors[x], vectors[y])
        bar = "#" * int(sim * 40)
        print(f"  Similarity({x} <-> {y}) = {sim:.4f}  {bar}")
        print(f"    {sentences[x][:55]}")
        print(f"    {sentences[y][:55]}")
        print()

    # ── Step 3: Word analogy demo ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Even single words become vectors")
    print("=" * 60)

    words = ["king", "queen", "man", "woman", "dog", "cat", "pizza"]
    word_vectors = {w: embed_text(w) for w in words}

    print()
    for w1 in words[:5]:
        for w2 in words[:5]:
            if w1 >= w2:
                continue
            sim = cosine_similarity(word_vectors[w1], word_vectors[w2])
            print(f"  {w1:8s} <-> {w2:8s}  ->  {sim:.4f}")

    print()
    print("Notice that  king<->queen  and  man<->woman  score higher")
    print("than  king<->pizza  -- the model understands relationships.")
