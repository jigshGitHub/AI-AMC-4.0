# ============================================================
#  SCRIPT 04 — Visualising Embeddings in 2D
#
#  1536-dimensional vectors are impossible to picture directly.
#  PCA compresses them to 2D so we can plot them.
#
#  You will see:
#    - Docs about similar topics cluster TOGETHER
#    - Docs about different topics are far APART
#    - This is exactly how a vector DB finds relevant docs!
# ============================================================

import os
import sys
import chromadb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from envsettings import getChromaDBDir

COLORS = {
    "Space":"#4C72B0",   # blue
    "Animals":"#55A868",   # green
    "Cooking":"#C44E52",   # red
    "Technology":"#DD8452",   # orange
}


if __name__ == "__main__":
    if os.system("cls" if os.name == "nt" else "clear") is not None:
        pass  # Clear the console for better readability

    chroma_db_dir = getChromaDBDir()
    collection_name = "EMBEDDINGS_COLLECTION"

    db = chromadb.PersistentClient(path=chroma_db_dir)
    collection = db.get_or_create_collection(name=collection_name)

    all_data = collection.get(include=["embeddings", "metadatas", "documents"])

    ids        = all_data["ids"]
    embeddings = np.array(all_data["embeddings"])   # shape: (20, 1536)
    metadatas  = all_data["metadatas"]
    documents  = all_data["documents"]

    print("=" * 60)
    print("EMBEDDING VISUALISER")
    print("=" * 60)
    print(f"\nLoaded {len(ids)} documents")
    n_docs, n_dims = embeddings.shape
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"  → {n_docs} documents × {n_dims} dimensions")


    # ── PCA: reduce 1536D → 2D ──────────────────────────────────
    print(f"\nRunning PCA to compress {n_dims} dimensions → 2 ...")
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)   # shape: (20, 2)

    var_explained = pca.explained_variance_ratio_
    print(f"Variance explained by 2 components: "
        f"{var_explained[0]*100:.1f}% + {var_explained[1]*100:.1f}% = "
        f"{sum(var_explained)*100:.1f}%")
    print("(Some information is lost — but clusters are still visible)")


    # ── Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Document Embeddings Visualised in 2D  (via PCA)", fontsize=15, fontweight="bold")


    # ── Left plot: categories colour-coded ──────────────────────
    ax1 = axes[0]
    ax1.set_title("Colour by Category", fontsize=12)

    categories = [m["category"] for m in metadatas]
    for i, (x, y) in enumerate(coords):
        cat   = categories[i]
        color = COLORS[cat]
        ax1.scatter(x, y, color=color, s=120, zorder=3, edgecolors="white", linewidths=0.8)

    # Annotate each point with a short label
    for i, (x, y) in enumerate(coords):
        short_title = metadatas[i]["title"]
        short_title = short_title if len(short_title) < 20 else short_title[:18] + "…"
        ax1.annotate(
            short_title, (x, y),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, alpha=0.85,
        )

    # Legend
    patches = [mpatches.Patch(color=c, label=cat) for cat, c in COLORS.items()]
    ax1.legend(handles=patches, loc="upper right", fontsize=9)
    ax1.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
    ax1.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f9f9f9")


    # ── Right plot: same but with big coloured category labels ──
    ax2 = axes[1]
    ax2.set_title("Cluster View (Labels Only)", fontsize=12)

    for i, (x, y) in enumerate(coords):
        cat   = categories[i]
        color = COLORS[cat]
        ax2.scatter(x, y, color=color, s=200, zorder=3, alpha=0.7, edgecolors="white", linewidths=1)
        ax2.text(x, y, cat[0], ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

    # Draw convex hull per category to highlight clusters
    from scipy.spatial import ConvexHull
    for cat, color in COLORS.items():
        indices = [i for i, m in enumerate(metadatas) if m["category"] == cat]
        pts = coords[indices]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(pts[hull.vertices], [pts[hull.vertices[0]]], axis=0)
                ax2.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.12)
                ax2.plot(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.4, linewidth=1.5)
            except Exception:
                pass

    ax2.legend(handles=patches, loc="upper right", fontsize=9)
    ax2.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
    ax2.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f9f9f9")


    plt.tight_layout()
    plt.savefig("embedding_clusters.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → embedding_clusters.png")
    plt.show()


    # ── Print 2D coordinates table ───────────────────────────────
    print("\n── 2D coordinates after PCA ─────────────────────────────")
    print(f"\n  {'ID':15s}  {'Category':12s}  {'PC1':>10}  {'PC2':>10}")
    print("  " + "-" * 52)
    for i, doc_id in enumerate(ids):
        x, y = coords[i]
        cat  = metadatas[i]["category"]
        print(f"  {doc_id:15s}  {cat:12s}  {x:>10.4f}  {y:>10.4f}")

    print("\nKey insight:")
    print("  Documents with similar MEANING end up close in space.")
    print("  A vector DB query works by finding the nearest neighbours")
    print("  to the query vector — exactly what you see in these clusters.")
