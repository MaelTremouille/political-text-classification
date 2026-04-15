"""
Frozen sentence-CamemBERT encoder + linear head classifier.

Motivation: fine-tuning CamemBERT (110M params) on 3,163 training examples
overfits. A frozen encoder + linear head has ~4k trainable parameters, which
is a proper fit for the data regime. Comparing it to TF-IDF + LogReg is
now an equal-capacity comparison of representations.

Uses pre-computed embeddings from data/document_embeddings.npy (produced by
political_mapping.py using dangvantuan/sentence-camembert-large). Alignment
with the parquets is by row order in corpus_labeled.parquet.

Protocols: held-out test (with bootstrap CI), 5-fold CV on train+val,
temporal splits. Mirrors src/evaluation.py so numbers are directly comparable.

Usage:
    python src/frozen_classifier.py \
        --embeddings data/document_embeddings.npy \
        --out data/results_frozen_sentence_camembert.txt
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
TARGET = "party_family"
SEED = 42
N_BOOTSTRAP = 1000


def load_aligned_data(emb_path: Path):
    """Load embeddings and the labeled parquet, in the same row order.

    document_embeddings.npy was produced by political_mapping.py iterating
    over data/corpus_labeled.parquet. So row i in the embeddings file maps
    to row i in that parquet.
    """
    df = pd.read_parquet(DATA_DIR / "corpus_labeled.parquet").reset_index(drop=True)
    emb = np.load(emb_path)
    if len(emb) != len(df):
        raise RuntimeError(
            f"Embedding count ({len(emb)}) != corpus_labeled rows ({len(df)}). "
            "Regenerate embeddings with political_mapping.py."
        )
    df["_emb_idx"] = np.arange(len(df))
    return df, emb


def reindex_split(split_df, corpus_df):
    """Map split rows back to their embedding index via doc_name."""
    idx = corpus_df.set_index("doc_name")["_emb_idx"]
    return idx.loc[split_df["doc_name"]].values


def fit_logreg(X, y):
    clf = LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=SEED, C=1.0
    )
    clf.fit(X, y)
    return clf


def bootstrap_macro_f1(y_true, y_pred, n=N_BOOTSTRAP, rng=None):
    rng = rng or np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def stratified_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        scaler = StandardScaler().fit(Xtr)
        clf = fit_logreg(scaler.transform(Xtr), ytr)
        yp = clf.predict(scaler.transform(Xva))
        rows.append({
            "fold": fold_idx,
            "acc": accuracy_score(yva, yp),
            "macro_f1": f1_score(yva, yp, average="macro", zero_division=0),
            "weighted_f1": f1_score(yva, yp, average="weighted", zero_division=0),
        })
    return pd.DataFrame(rows)


def temporal_split_eval(corpus_df, emb, early_max, late_min, label):
    years_int = corpus_df["year"].astype(int).values
    early_mask = years_int <= early_max
    late_mask = years_int >= late_min
    if not early_mask.any() or not late_mask.any():
        return None
    Xtr = emb[early_mask]
    ytr = corpus_df.loc[early_mask, TARGET].values
    Xte = emb[late_mask]
    yte = corpus_df.loc[late_mask, TARGET].values
    scaler = StandardScaler().fit(Xtr)
    clf = fit_logreg(scaler.transform(Xtr), ytr)
    yp = clf.predict(scaler.transform(Xte))

    per_year = {}
    late_years = corpus_df.loc[late_mask, "year"].values
    for yr in sorted(set(late_years)):
        m = late_years == yr
        per_year[yr] = {
            "n": int(m.sum()),
            "acc": float(accuracy_score(yte[m], yp[m])),
            "macro_f1": float(f1_score(yte[m], yp[m], average="macro", zero_division=0)),
        }

    return {
        "label": label,
        "train_years": sorted(set(corpus_df.loc[early_mask, "year"].values)),
        "test_years": sorted(set(late_years)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "acc": float(accuracy_score(yte, yp)),
        "macro_f1": float(f1_score(yte, yp, average="macro", zero_division=0)),
        "per_year": per_year,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/document_embeddings.npy")
    parser.add_argument("--out", default="data/results_frozen_sentence_camembert.txt")
    parser.add_argument("--preds_out", default="data/preds_frozen_scam.parquet")
    args = parser.parse_args()

    # Align corpus_labeled with embeddings, then map splits back
    corpus, emb = load_aligned_data(Path(args.embeddings))

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    tr_idx = reindex_split(train, corpus)
    va_idx = reindex_split(val, corpus)
    te_idx = reindex_split(test, corpus)

    Xtr, ytr = emb[tr_idx], train[TARGET].values
    Xva, yva = emb[va_idx], val[TARGET].values
    Xte, yte = emb[te_idx], test[TARGET].values

    # --- Protocol 1: held-out test ---
    scaler = StandardScaler().fit(Xtr)
    clf = fit_logreg(scaler.transform(Xtr), ytr)
    y_pred_val = clf.predict(scaler.transform(Xva))
    y_pred_test = clf.predict(scaler.transform(Xte))

    acc_test = accuracy_score(yte, y_pred_test)
    mf1_mean, mf1_lo, mf1_hi = bootstrap_macro_f1(yte, y_pred_test)

    # --- Protocol 2: 5-fold CV on train+val ---
    Xtrva = np.concatenate([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])
    cv_df = stratified_cv(Xtrva, ytrva, n_splits=5)

    # --- Protocol 3: temporal splits ---
    temp1 = temporal_split_eval(corpus, emb, 1981, 1988, "Train <=1981 / Test >=1988")
    temp2 = temporal_split_eval(corpus, emb, 1988, 1993, "Train <=1988 / Test ==1993")
    temp_rev = temporal_split_eval(corpus, emb, 1978, 1988, "Train <=1978 / Test >=1988")

    # --- Save test predictions for error analysis ---
    pd.DataFrame({
        "doc_name": test["doc_name"].values,
        "year": test["year"].values,
        "dept_code": test["dept_code"].values,
        "party_family": yte,
        "pred": y_pred_test,
    }).to_parquet(args.preds_out, index=False)

    # --- Report ---
    lines = []
    lines.append("=" * 70)
    lines.append(f"Frozen sentence-CamemBERT + LogReg  |  {datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"Embeddings: {args.embeddings}  (shape={emb.shape})")
    lines.append("=" * 70)
    lines.append("")
    lines.append("### Held-out test (stratified 70/15/15) ###")
    lines.append(f"Test accuracy          : {acc_test:.4f}")
    lines.append(f"Test macro-F1 (point)  : {f1_score(yte, y_pred_test, average='macro', zero_division=0):.4f}")
    lines.append(f"Bootstrap CI (macro-F1): {mf1_mean:.4f}  [{mf1_lo:.4f} ; {mf1_hi:.4f}]  (N={N_BOOTSTRAP})")
    lines.append("")
    lines.append("--- Test classification report ---")
    lines.append(classification_report(yte, y_pred_test, zero_division=0))
    labels = sorted(set(yte))
    cm = confusion_matrix(yte, y_pred_test, labels=labels)
    lines.append("Confusion matrix:")
    header = f"{'':>18s} " + " ".join(f"{l:>16s}" for l in labels)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"{labels[i]:>18s} " + " ".join(f"{v:>16d}" for v in row))
    lines.append("")

    lines.append("### Stratified 5-fold CV on (train+val) ###")
    lines.append(cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append(f"Mean acc      : {cv_df['acc'].mean():.4f} ± {cv_df['acc'].std():.4f}")
    lines.append(f"Mean macro-F1 : {cv_df['macro_f1'].mean():.4f} ± {cv_df['macro_f1'].std():.4f}")
    lines.append(f"Mean weighted : {cv_df['weighted_f1'].mean():.4f} ± {cv_df['weighted_f1'].std():.4f}")
    lines.append("")

    lines.append("### Temporal splits ###")
    for t in [temp1, temp2, temp_rev]:
        if t is None:
            continue
        lines.append(f"-- {t['label']} (n_train={t['n_train']}, n_test={t['n_test']}) --")
        lines.append(f"   acc={t['acc']:.4f}  macro-F1={t['macro_f1']:.4f}")
        for yr, s in t["per_year"].items():
            lines.append(f"   year={yr} n={s['n']:>4d}  acc={s['acc']:.4f}  macro-F1={s['macro_f1']:.4f}")
    lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
