"""
Complement to political_mapping.py:
- UMAP projection of document embeddings (second view alongside t-SNE).
- Clustering quality metrics (silhouette, Davies-Bouldin) on raw embeddings
  and both 2D projections, partitioned by political family.
- Per-party temporal trajectories in the embedding space (bonus).

Input:  data/corpus_labeled.parquet, data/document_embeddings.npy
Output: figures/umap_by_family.png, figures/party_trajectories.png,
        figures/clustering_metrics.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

FAMILY_COLORS = {
    "Extreme gauche": "red",
    "Gauche": "salmon",
    "Droite": "blue",
    "Extreme droite": "darkblue",
}

PARTY_COLORS = {
    # Extreme gauche
    "LO": "#8B0000", "LCR": "#B22222",
    # Gauche
    "PCF": "#DC143C", "PS": "#FF6347", "PSU": "#FA8072",
    "MRG": "#CD5C5C", "Union Gauche": "#E9967A",
    # Droite
    "UDF": "#4169E1", "RPR": "#1E90FF", "UDR": "#6495ED", "RI": "#87CEEB",
    "URC": "#5F9EA0", "UPF": "#4682B4", "Maj. Pres.": "#00008B",
    "CNI": "#B0C4DE", "RPCR": "#6A5ACD",
    # Extreme droite
    "FN": "#191970", "PFN": "#000080",
}


def main():
    df = pd.read_parquet("data/corpus_labeled.parquet").reset_index(drop=True)
    emb = np.load("data/document_embeddings.npy")
    assert len(emb) == len(df), "Mismatch between embeddings and corpus"

    # --- 1. UMAP ---
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42,
                        metric="cosine")
    emb_umap = reducer.fit_transform(emb)

    plt.figure(figsize=(12, 8))
    for family, color in FAMILY_COLORS.items():
        mask = df["party_family"] == family
        plt.scatter(emb_umap[mask, 0], emb_umap[mask, 1], c=color,
                    label=family, alpha=0.4, s=10)
    plt.legend()
    plt.title("UMAP projection of professions de foi by political family")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(OUT / "umap_by_family.png", dpi=150)
    plt.close()
    print("Saved umap_by_family.png")

    # --- 2. Clustering metrics (silhouette, Davies-Bouldin) by family ---
    # Run fresh t-SNE to match the existing figure
    print("Running t-SNE for metrics...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_tsne = tsne.fit_transform(emb)

    labels = df["party_family"].values

    metrics = {}
    for name, X in [("raw (1024-d)", emb), ("t-SNE 2D", emb_tsne), ("UMAP 2D", emb_umap)]:
        sil = silhouette_score(X, labels, metric="cosine" if name == "raw (1024-d)" else "euclidean")
        db = davies_bouldin_score(X, labels)
        metrics[name] = {"silhouette": sil, "davies_bouldin": db}

    lines = ["Clustering quality by political family", "=" * 60, ""]
    lines.append(f"{'Representation':<20s}  {'Silhouette':>12s}  {'Davies-Bouldin':>16s}")
    lines.append("-" * 60)
    for name, m in metrics.items():
        lines.append(f"{name:<20s}  {m['silhouette']:>12.4f}  {m['davies_bouldin']:>16.4f}")
    lines.append("")
    lines.append("Silhouette closer to 1 = tight clusters; Davies-Bouldin lower = better separated.")
    (OUT / "clustering_metrics.txt").write_text("\n".join(lines) + "\n")
    print("Saved clustering_metrics.txt")
    print("\n".join(lines))

    # --- 3. Per-party temporal trajectories (bonus) ---
    print("\nBuilding per-party temporal trajectories...")
    years = sorted(df["year"].astype(int).unique())
    party_year_mean = {}
    party_counts = df["party_label"].value_counts()
    # Only parties with enough docs (>=10) and at least 2 years of data
    eligible_parties = []
    for party in party_counts.index:
        present_years = sorted(df[df["party_label"] == party]["year"].astype(int).unique())
        if party_counts[party] >= 10 and len(present_years) >= 2:
            eligible_parties.append(party)
    print(f"  {len(eligible_parties)} parties eligible for trajectories: {eligible_parties}")

    # Project the means into UMAP space for a consistent visualisation
    means = []
    keys = []
    for party in eligible_parties:
        for yr in sorted(df[df["party_label"] == party]["year"].astype(int).unique()):
            mask = (df["party_label"] == party) & (df["year"].astype(int) == yr)
            if mask.sum() >= 3:  # need at least 3 docs for a stable mean
                means.append(emb[mask].mean(axis=0))
                keys.append((party, yr))
    means_arr = np.array(means)
    # Re-use the fitted UMAP to transform the means
    means_2d = reducer.transform(means_arr)

    plt.figure(figsize=(13, 9))
    plotted_labels = set()
    for party in eligible_parties:
        pts = [(yr, means_2d[i]) for i, (p, yr) in enumerate(keys) if p == party]
        pts.sort(key=lambda x: x[0])
        if not pts:
            continue
        xs = [p[1][0] for p in pts]
        ys = [p[1][1] for p in pts]
        color = PARTY_COLORS.get(party, "grey")
        lbl = party if party not in plotted_labels else None
        plotted_labels.add(party)
        plt.plot(xs, ys, marker="o", color=color, linewidth=1.6,
                 markersize=5, alpha=0.85, label=lbl)
        # Annotate first and last year
        plt.annotate(f"{party} {pts[0][0]}", (xs[0], ys[0]),
                     fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        plt.annotate(f"{party} {pts[-1][0]}", (xs[-1], ys[-1]),
                     fontsize=7, alpha=0.9, xytext=(3, 3), textcoords="offset points")

    plt.title("Per-party discourse trajectories, 1973-1993 (UMAP of mean embeddings)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "party_trajectories.png", dpi=150)
    plt.close()
    print("Saved party_trajectories.png")


if __name__ == "__main__":
    main()
