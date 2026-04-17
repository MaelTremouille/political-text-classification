"""
Re-run the TF-IDF + logistic regression pipeline with a bit more rigour than
the baseline classification.py script:
- held-out test with a bootstrap CI on macro-F1,
- stratified 5-fold CV on train+val,
- a few temporal splits (train on early years, test on later, and back).

Also dumps the top-20 features per class and the test predictions so the
error-analysis script can pick them up. We take --text_col as an argument
so we can re-run on text_clean (baseline) or text_clean_v2 (cleaned).

Usage:
    python src/evaluation.py --text_col text_clean_v2 --out data/results_tfidf_v2.txt
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("data")
TARGET = "party_family"
SEED = 42
N_BOOTSTRAP = 1000


def load_splits():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    return train, val, test


def fit_tfidf_logreg(train_texts, train_labels):
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
    X = vect.fit_transform(train_texts)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED
    )
    clf.fit(X, train_labels)
    return vect, clf


def predict(vect, clf, texts):
    return clf.predict(vect.transform(texts))


def bootstrap_macro_f1(y_true, y_pred, n=N_BOOTSTRAP, rng=None):
    """Bootstrap CI on macro F1 given fixed predictions."""
    rng = rng or np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores[i] = f1_score(
            y_true[idx], y_pred[idx], average="macro", zero_division=0
        )
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def stratified_cv(df, text_col, n_splits=5):
    """5-fold stratified CV on train+val: refit TF-IDF inside each fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    per_fold = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df[TARGET])):
        tr = df.iloc[train_idx]
        va = df.iloc[val_idx]
        vect, clf = fit_tfidf_logreg(tr[text_col].values, tr[TARGET].values)
        y_pred = predict(vect, clf, va[text_col].values)
        acc = accuracy_score(va[TARGET], y_pred)
        mf1 = f1_score(va[TARGET], y_pred, average="macro", zero_division=0)
        wf1 = f1_score(va[TARGET], y_pred, average="weighted", zero_division=0)
        per_fold.append({"fold": fold_idx, "acc": acc, "macro_f1": mf1, "weighted_f1": wf1})
    return pd.DataFrame(per_fold)


def top_features_per_class(vect, clf, top_k=20):
    """Top K features per class by coef * mean(idf)."""
    vocab = np.array(vect.get_feature_names_out())
    idf = vect.idf_
    rows = []
    for i, cls in enumerate(clf.classes_):
        scores = clf.coef_[i] * idf
        top_idx = np.argsort(-scores)[:top_k]
        for rank, j in enumerate(top_idx):
            rows.append({
                "class": cls,
                "rank": rank + 1,
                "feature": vocab[j],
                "score": float(scores[j]),
                "coef": float(clf.coef_[i, j]),
                "idf": float(idf[j]),
            })
    return pd.DataFrame(rows)


def temporal_split_eval(all_df, text_col, early_max, late_min, label):
    """Train on rows with year <= early_max, test on year >= late_min."""
    early = all_df[all_df["year"].astype(int) <= early_max]
    late = all_df[all_df["year"].astype(int) >= late_min]
    if len(early) == 0 or len(late) == 0:
        return None
    vect, clf = fit_tfidf_logreg(early[text_col].values, early[TARGET].values)
    y_pred = predict(vect, clf, late[text_col].values)
    acc = accuracy_score(late[TARGET], y_pred)
    mf1 = f1_score(late[TARGET], y_pred, average="macro", zero_division=0)
    per_year = {}
    for yr in sorted(late["year"].unique()):
        mask = late["year"] == yr
        yt = late[mask][TARGET].values
        yp = y_pred[mask.values]
        if len(yt) == 0:
            continue
        per_year[yr] = {
            "n": int(len(yt)),
            "acc": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        }
    return {
        "label": label,
        "train_years": sorted(early["year"].unique().tolist()),
        "test_years": sorted(late["year"].unique().tolist()),
        "n_train": int(len(early)),
        "n_test": int(len(late)),
        "acc": float(acc),
        "macro_f1": float(mf1),
        "per_year": per_year,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_col", default="text_clean_v2")
    parser.add_argument("--out", default="data/results_tfidf_v2.txt")
    parser.add_argument("--top_features_csv", default=None)
    parser.add_argument("--preds_out", default=None,
                        help="Save test predictions (doc_name, true, pred) for error analysis")
    args = parser.parse_args()

    train, val, test = load_splits()
    all_df = pd.concat([train, val, test], ignore_index=True)

    # --- Protocol 1: held-out test (same as baseline) ---
    vect, clf = fit_tfidf_logreg(train[args.text_col].values, train[TARGET].values)
    y_pred_test = predict(vect, clf, test[args.text_col].values)
    y_pred_val = predict(vect, clf, val[args.text_col].values)

    acc_test = accuracy_score(test[TARGET], y_pred_test)
    mf1_mean, mf1_lo, mf1_hi = bootstrap_macro_f1(test[TARGET].values, y_pred_test)

    # --- Protocol 2: 5-fold CV on train+val ---
    trainval = pd.concat([train, val], ignore_index=True)
    cv_df = stratified_cv(trainval, args.text_col, n_splits=5)

    # --- Protocol 3: temporal splits ---
    temp1 = temporal_split_eval(all_df, args.text_col, early_max=1981, late_min=1988,
                                label="Train <=1981 / Test >=1988")
    temp2 = temporal_split_eval(all_df, args.text_col, early_max=1988, late_min=1993,
                                label="Train <=1988 / Test ==1993")
    temp_rev = temporal_split_eval(all_df, args.text_col, early_max=1978, late_min=1988,
                                   label="Train <=1978 / Test >=1988")

    # --- Top features ---
    top_feats = top_features_per_class(vect, clf, top_k=20)
    if args.top_features_csv:
        top_feats.to_csv(args.top_features_csv, index=False)
    elif args.text_col:
        auto_csv = f"data/tfidf_top_features_{args.text_col.replace('text_clean', '').lstrip('_') or 'v1'}.csv"
        top_feats.to_csv(auto_csv, index=False)

    # --- Save predictions for error analysis ---
    preds_out = args.preds_out or f"data/preds_tfidf_{args.text_col.replace('text_clean', '').lstrip('_') or 'v1'}.parquet"
    pd.DataFrame({
        "doc_name": test["doc_name"].values,
        "year": test["year"].values,
        "dept_code": test["dept_code"].values,
        "party_family": test[TARGET].values,
        "pred": y_pred_test,
    }).to_parquet(preds_out, index=False)

    # --- Write report ---
    lines = []
    lines.append("=" * 70)
    lines.append(f"TF-IDF + LogReg  |  text_col={args.text_col}  |  {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("### Held-out test (stratified 70/15/15, seed=42) ###")
    lines.append(f"Test accuracy          : {acc_test:.4f}")
    lines.append(f"Test macro-F1 (point)  : {f1_score(test[TARGET], y_pred_test, average='macro', zero_division=0):.4f}")
    lines.append(f"Bootstrap CI (macro-F1): {mf1_mean:.4f}  [{mf1_lo:.4f} ; {mf1_hi:.4f}]  (N={N_BOOTSTRAP})")
    lines.append("")
    lines.append("--- Test classification report ---")
    lines.append(classification_report(test[TARGET], y_pred_test, zero_division=0))
    lines.append("")
    labels = sorted(test[TARGET].unique())
    cm = confusion_matrix(test[TARGET], y_pred_test, labels=labels)
    lines.append("Confusion matrix (rows = true, cols = predicted):")
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

    lines.append("### Top 20 features per class (coef * idf) ###")
    for cls, g in top_feats.groupby("class"):
        lines.append(f"--- {cls} ---")
        for _, r in g.iterrows():
            lines.append(f"  {r['rank']:2d}. {r['feature']:<28s}  score={r['score']:+.3f}  coef={r['coef']:+.3f}  idf={r['idf']:.2f}")
        lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
