import os
import sys
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def getOpenAIClient():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def getembeddingmodel():
    return os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-small")

def embed_text(text):
    client = getOpenAIClient()
    response = client.embeddings.create(
        input=text,
        model=getembeddingmodel()
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
