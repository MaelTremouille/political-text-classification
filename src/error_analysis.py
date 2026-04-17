"""
Quick error analysis: where do TF-IDF v2 and the frozen sentence-CamemBERT
head succeed / fail? We slice accuracy by document length, by election year
and by political family, and we print a few documents where the two models
disagree so we can look at them by hand.

Usage:
    python src/error_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
FIG_DIR = Path("figures/error_analysis")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def add_doc_features(preds_df, corpus):
    """Attach the cleaned text and the document length (in tokens)."""
    meta = corpus.set_index("doc_name")[["text_clean", "year", "party_family"]]
    preds_df = preds_df.merge(
        meta.rename(columns={"text_clean": "_text"}),
        left_on="doc_name", right_index=True, how="left", suffixes=("", "_meta"),
    )
    preds_df["n_tokens"] = preds_df["_text"].str.split().str.len()
    return preds_df


def bucketise(x, edges, labels):
    return pd.cut(x, bins=edges, labels=labels, include_lowest=True)


def acc_by_group(df, group_col):
    df = df.copy()
    df["correct"] = (df["pred"] == df["party_family"]).astype(int)
    return df.groupby(group_col).agg(n=("correct", "size"), acc=("correct", "mean")).round(3)


def plot_acc_by(tfidf, scam, key, fig_path, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, d, marker in [("TF-IDF (v2)", tfidf, "o"), ("sentence-CamemBERT", scam, "s")]:
        g = acc_by_group(d, key)
        ax.plot(g.index.astype(str), g["acc"], marker=marker, label=name)
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    tfidf = pd.read_parquet(DATA_DIR / "preds_tfidf_v2.parquet")
    scam = pd.read_parquet(DATA_DIR / "preds_frozen_scam.parquet")
    corpus = pd.read_parquet(DATA_DIR / "corpus_labeled.parquet")

    tfidf = add_doc_features(tfidf, corpus)
    scam = add_doc_features(scam, corpus)

    # Length buckets
    edges = [0, 500, 1500, 3000, 100000]
    labels = ["0-500", "500-1500", "1500-3000", "3000+"]
    tfidf["len_bucket"] = bucketise(tfidf["n_tokens"], edges, labels)
    scam["len_bucket"] = bucketise(scam["n_tokens"], edges, labels)

    # Tables
    report = []
    report.append("=" * 60)
    report.append("Error analysis (TF-IDF v2 vs frozen sentence-CamemBERT)")
    report.append("=" * 60 + "\n")
    for key, title in [
        ("year", "By election year"),
        ("party_family", "By political family"),
        ("len_bucket", "By document length (tokens)"),
    ]:
        report.append(f"--- {title} ---")
        report.append("TF-IDF v2:")
        report.append(acc_by_group(tfidf, key).to_string())
        report.append("")
        report.append("sentence-CamemBERT:")
        report.append(acc_by_group(scam, key).to_string())
        report.append("\n")

    # Disagreements
    both = tfidf[["doc_name", "party_family", "pred"]].merge(
        scam[["doc_name", "pred"]], on="doc_name", suffixes=("_tfidf", "_scam")
    )
    disagree = both[both["pred_tfidf"] != both["pred_scam"]]
    report.append(f"Disagreements (TF-IDF ≠ CamemBERT): {len(disagree)} / {len(both)} "
                  f"({100*len(disagree)/len(both):.1f}%)")
    both_wrong = both[(both["pred_tfidf"] != both["party_family"]) &
                      (both["pred_scam"] != both["party_family"])]
    report.append(f"Both models wrong: {len(both_wrong)} / {len(both)} "
                  f"({100*len(both_wrong)/len(both):.1f}%)")
    report.append("")

    # Qualitative examples (up to 15): disagreement cases
    if len(disagree) > 0:
        report.append("--- 15 disagreement examples (doc, true, tfidf, scam) ---")
        meta = corpus.set_index("doc_name")["text_clean"]
        sample = disagree.head(15)
        for _, row in sample.iterrows():
            snippet = meta.get(row["doc_name"], "")[:200].replace("\n", " ")
            report.append(f"[{row['doc_name']}]  true={row['party_family']:<16s}"
                          f"  tfidf={row['pred_tfidf']:<16s}  scam={row['pred_scam']:<16s}")
            report.append(f"    '{snippet}'")

    out = FIG_DIR / "error_analysis.txt"
    out.write_text("\n".join(report) + "\n")
    print(f"Wrote {out}")

    # Figures (we dropped the OCR-quality plot: the non-alpha ratio proxy
    # was not discriminative on our corpus, almost everything sat below 10%)
    plot_acc_by(tfidf, scam, "year", FIG_DIR / "acc_by_year.png",
                "Accuracy per election year")
    plot_acc_by(tfidf, scam, "len_bucket", FIG_DIR / "acc_by_length.png",
                "Accuracy per document length bucket")

    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
