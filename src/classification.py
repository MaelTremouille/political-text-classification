"""
Train and evaluate two classifiers on the professions de foi corpus:
1) TF-IDF + logistic regression (baseline)
2) Fine-tuned CamemBERT

Results are printed and appended to data/results.txt.
The CamemBERT model is saved to data/camembert_classifier/.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load splits
train_df = pd.read_parquet("data/train.parquet")
val_df = pd.read_parquet("data/val.parquet")
test_df = pd.read_parquet("data/test.parquet")

TARGET = "party_family"

# --- Baseline: TF-IDF + Logistic Regression ---
print("=== Baseline: TF-IDF + Logistic Regression ===\n")

# TF-IDF: up to 20k features, unigrams + bigrams, min 3 docs per term
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
X_train = tfidf.fit_transform(train_df["text_clean"])
X_val = tfidf.transform(val_df["text_clean"])
X_test = tfidf.transform(test_df["text_clean"])

y_train = train_df[TARGET]
y_val = val_df[TARGET]
y_test = test_df[TARGET]

# Train
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Evaluate on validation set
y_pred_val = clf.predict(X_val)
print("--- Validation set ---")
print(classification_report(y_val, y_pred_val))

# Evaluate on test set
y_pred_test = clf.predict(X_test)
print("--- Test set ---")
print(classification_report(y_test, y_pred_test))

print("Confusion matrix (test):")
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred_test, labels=labels)
print(f"{'':>20s}", "  ".join(f"{l:>15s}" for l in labels))
for i, row in enumerate(cm):
    print(f"{labels[i]:>20s}", "  ".join(f"{v:>15d}" for v in row))


# === CamemBERT Fine-tuning ===
print("\n\n=== CamemBERT Fine-tuning ===\n")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Encode labels as integers
label_list = sorted(train_df[TARGET].unique())
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label_list)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("camembert-base")


class ProfessionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx], max_length=512, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx]),
        }


print("Building datasets...")
train_dataset = ProfessionDataset(train_df["text_clean"].values, train_df[TARGET].values)
val_dataset = ProfessionDataset(val_df["text_clean"].values, val_df[TARGET].values)
test_dataset = ProfessionDataset(test_df["text_clean"].values, test_df[TARGET].values)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "camembert-base", num_labels=num_labels,
)
model.to(DEVICE)

# Class weights to handle imbalance (Extreme droite has very few examples)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight("balanced", classes=np.array(range(num_labels)), y=[label2id[l] for l in train_df[TARGET]])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
print(f"Class weights: {dict(zip(label_list, class_weights.cpu().numpy().round(2)))}")

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
NUM_EPOCHS = 5  # overfitting starts around epoch 3, but we log everything

from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx + 1}/{num_batches}")

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

    val_acc = correct / total
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} — loss: {avg_loss:.4f} — val accuracy: {val_acc:.4f}")

# Test evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

y_pred_bert = [id2label[p] for p in all_preds]
y_true_bert = [id2label[l] for l in all_labels]

print("\n--- CamemBERT Test set ---")
print(classification_report(y_true_bert, y_pred_bert))

print("Confusion matrix (test):")
cm = confusion_matrix(y_true_bert, y_pred_bert, labels=label_list)
print(f"{'':>20s}", "  ".join(f"{l:>15s}" for l in label_list))
for i, row in enumerate(cm):
    print(f"{label_list[i]:>20s}", "  ".join(f"{v:>15d}" for v in row))

# Save model
model.save_pretrained("data/camembert_classifier")
tokenizer.save_pretrained("data/camembert_classifier")
print("\nModel saved to data/camembert_classifier")

# Save all results to file
from datetime import datetime
with open("data/results.txt", "a") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"{'='*60}\n\n")
    f.write("--- TF-IDF Baseline (test) ---\n")
    f.write(classification_report(y_test, y_pred_test))
    f.write("\n--- CamemBERT (test) ---\n")
    f.write(classification_report(y_true_bert, y_pred_bert))
    f.write("\n")
print("Results appended to data/results.txt")
