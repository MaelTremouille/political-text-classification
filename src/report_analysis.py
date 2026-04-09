"""
Generate additional statistics for the report:
1) Text length (word count) by political family
2) Top 15 most discriminative TF-IDF features per class
3) Confusion matrix for TF-IDF + Logistic Regression

This script does not retrain CamemBERT. It only fits a TF-IDF + LogReg
(takes a few seconds) to extract feature coefficients.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# --- Load data ---

corpus = pd.read_parquet("data/corpus_labeled.parquet")
train_df = pd.read_parquet("data/train.parquet")
test_df = pd.read_parquet("data/test.parquet")

TARGET = "party_family"

# =====================================================================
# 1) Text length statistics by political family
# =====================================================================

print("=" * 60)
print("TEXT LENGTH STATISTICS (word count)")
print("=" * 60)

corpus["word_count"] = corpus["text_clean"].str.split().str.len()

stats = corpus.groupby(TARGET)["word_count"].agg(["mean", "median", "std", "count"])
stats = stats.round(0).astype(int)
print(stats.to_string())
print()

# Overall stats
print(f"Overall: mean={corpus['word_count'].mean():.0f}, "
      f"median={corpus['word_count'].median():.0f}, "
      f"std={corpus['word_count'].std():.0f}")
print()

# =====================================================================
# 2) Top TF-IDF features per class (logistic regression coefficients)
# =====================================================================

print("=" * 60)
print("TOP 20 TF-IDF FEATURES PER CLASS")
print("=" * 60)

# Same pipeline as classification.py
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
X_train = tfidf.fit_transform(train_df["text_clean"])
X_test = tfidf.transform(test_df["text_clean"])

y_train = train_df[TARGET]
y_test = test_df[TARGET]

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

feature_names = np.array(tfidf.get_feature_names_out())

# For each class, get the top 15 features by coefficient value
for i, label in enumerate(clf.classes_):
    coefs = clf.coef_[i]
    top_indices = np.argsort(coefs)[-20:][::-1]
    top_features = feature_names[top_indices]
    top_values = coefs[top_indices]

    print(f"\n--- {label} ---")
    for feat, val in zip(top_features, top_values):
        print(f"  {feat:30s}  {val:.3f}")

print()

# =====================================================================
# 3) Confusion matrix (TF-IDF baseline)
# =====================================================================

print("=" * 60)
print("CONFUSION MATRIX — TF-IDF + Logistic Regression (test set)")
print("=" * 60)

y_pred = clf.predict(X_test)
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Print with aligned labels
header = f"{'Predicted ->':>20s}  " + "  ".join(f"{l:>15s}" for l in labels)
print(header)
print("-" * len(header))
for i, row in enumerate(cm):
    print(f"{labels[i]:>20s}  " + "  ".join(f"{v:>15d}" for v in row))

print()
print("Rows = true label, Columns = predicted label")
