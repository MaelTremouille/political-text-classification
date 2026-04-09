"""
Compute sentence embeddings for the professions de foi corpus and produce
three visualizations:
- t-SNE projection colored by political family
- Cosine similarity heatmap between parties
- Temporal evolution of political families in embedding space

We use dangvantuan/sentence-camembert-large (a sentence-transformer built
on CamemBERT-large). We tried CamemBERT base [CLS] embeddings first but
cosine similarities were all >0.98 (useless), see figures/*_base.png.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: Extract embeddings ---

df = pd.read_parquet("data/corpus_labeled.parquet")

model = SentenceTransformer("dangvantuan/sentence-camembert-large")

print("Extracting embeddings...")
embeddings = model.encode(
    df["text_clean"].tolist(),
    show_progress_bar=True,
    batch_size=32,
)

np.save("data/document_embeddings.npy", embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# --- Step 2: Mean embeddings per party and per party-year ---

party_embeddings = {}
for party in df["party_label"].unique():
    mask = df["party_label"] == party
    party_embeddings[party] = embeddings[mask].mean(axis=0)

family_embeddings = {}
for family in df["party_family"].unique():
    mask = df["party_family"] == family
    family_embeddings[family] = embeddings[mask].mean(axis=0)

# --- Step 3: t-SNE projection colored by party family ---

print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

FAMILY_COLORS = {
    "Extreme gauche": "red",
    "Gauche": "salmon",
    "Ecologistes": "green",
    "Droite": "blue",
    "Extreme droite": "darkblue",
}

plt.figure(figsize=(12, 8))
for family, color in FAMILY_COLORS.items():
    mask = df["party_family"] == family
    plt.scatter(
        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
        c=color, label=family, alpha=0.4, s=10,
    )
plt.legend()
plt.title("t-SNE projection of professions de foi by political family")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tsne_by_family.png", dpi=150)
plt.close()
print("Saved tsne_by_family.png")

# --- Step 4: Cosine similarity heatmap between parties ---

party_names = sorted(party_embeddings.keys())
party_matrix = np.array([party_embeddings[p] for p in party_names])
sim_matrix = cosine_similarity(party_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(
    sim_matrix, xticklabels=party_names, yticklabels=party_names,
    annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
)
plt.title("Cosine similarity between parties (sentence-transformer embeddings)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cosine_similarity_parties.png", dpi=150)
plt.close()
print("Saved cosine_similarity_parties.png")

# --- Step 5: Temporal evolution ---
# One point per (family, year) pair — only 18 points so t-SNE is limited

years = sorted(df["year"].unique())
families = sorted(FAMILY_COLORS.keys())

family_year_embeddings = {}
for family in families:
    for year in years:
        mask = (df["party_family"] == family) & (df["year"] == year)
        if mask.sum() > 0:
            family_year_embeddings[(family, year)] = embeddings[mask].mean(axis=0)

# t-SNE on family-year mean embeddings
fy_labels = list(family_year_embeddings.keys())
fy_matrix = np.array([family_year_embeddings[k] for k in fy_labels])
tsne_fy = TSNE(n_components=2, random_state=42, perplexity=min(5, len(fy_labels) - 1))
fy_2d = tsne_fy.fit_transform(fy_matrix)

plt.figure(figsize=(12, 8))
for i, (family, year) in enumerate(fy_labels):
    color = FAMILY_COLORS[family]
    plt.scatter(fy_2d[i, 0], fy_2d[i, 1], c=color, s=80)
    plt.annotate(f"{family[:3]} {year}", (fy_2d[i, 0], fy_2d[i, 1]), fontsize=7)

# Legend
for family, color in FAMILY_COLORS.items():
    plt.scatter([], [], c=color, label=family)
plt.legend()
plt.title("Political families over time (t-SNE of mean embeddings)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tsne_temporal_evolution.png", dpi=150)
plt.close()
print("Saved tsne_temporal_evolution.png")

print("\nDone!")
