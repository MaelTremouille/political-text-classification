import re
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/corpus_by_document.parquet")
OUTPUT_PATH = Path("data/corpus_labeled.parquet")

df = pd.read_parquet(INPUT_PATH)

# Party patterns: (regex, label)
# Order matters: more specific patterns first to avoid wrong matches
PARTY_PATTERNS = [
    # Extreme left
    (r"LUTTE\s+OUVRI[EÈ]RE", "LO"),
    (r"LIGUE\s+COMMUNISTE\s+R[EÉ]VOLUTIONNAIRE", "LCR"),
    (r"LIGUE\s+COMMUNISTE", "LCR"),

    # Left
    (r"PARTI\s+COMMUNISTE\s+FRAN[CÇ]AIS", "PCF"),
    (r"PARTI\s+SOCIALISTE\s+UNIFI[EÉ]", "PSU"),
    (r"MOUVEMENT\s+DES\s+RADICAUX\s+DE\s+GAUCHE", "MRG"),
    (r"RADICAUX\s+DE\s+GAUCHE", "MRG"),
    (r"PARTI\s+SOCIALISTE", "PS"),

    # Center / Right
    (r"UNION\s+POUR\s+LA\s+D[EÉ]MOCRATIE\s+FRAN[CÇ]AISE", "UDF"),
    (r"CENTRE\s+DES\s+D[EÉ]MOCRATES\s+SOCIAUX", "UDF"),
    (r"PARTI\s+R[EÉ]PUBLICAIN", "UDF"),
    (r"R[EÉ]PUBLICAINS?\s+IND[EÉ]PENDANTS?", "RI"),
    (r"RASSEMBLEMENT\s+POUR\s+LA\s+R[EÉ]PUBLIQUE", "RPR"),
    (r"UNION\s+DES\s+D[EÉ]MOCRATES\s+POUR\s+LA\s+R[EÉ]PUBLIQUE", "UDR"),
    (r"UNION\s+DU\s+RASSEMBLEMENT\s+ET\s+DU\s+CENTRE", "URC"),
    (r"UNION\s+POUR\s+LA\s+FRANCE", "UPF"),
    (r"MAJORIT[EÉ]\s+PR[EÉ]SIDENTIELLE", "Maj. Pres."),

    # Far right
    (r"FRONT\s+NATIONAL", "FN"),
    (r"PARTI\s+DES\s+FORCES\s+NOUVELLES", "PFN"),

    # Ecology
    (r"LES\s+VERTS", "Verts"),
    (r"[EÉ]COLOGISTE", "Verts"),

    # Left coalitions
    (r"CANDIDATS?\s+SOCIALISTES?", "PS"),
    (r"UNION\s+DE\s+LA\s+GAUCHE", "Union Gauche"),
    (r"RASSEMBLEMENT\s+DES\s+FORCES\s+DE\s+GAUCHE", "Union Gauche"),
]

# Short abbreviations — only matched in the header (first 1000 chars)
# to avoid false positives deeper in the text
ABBREV_PATTERNS = [
    (r"\bP\.?C\.?F\.?\b", "PCF"),
    (r"\bP\.?S\.?U\.?\b", "PSU"),
    (r"\bP\.?S\.?\b", "PS"),
    (r"\bR\.?P\.?R\.?\b", "RPR"),
    (r"\bU\.?D\.?F\.?\b", "UDF"),
    (r"\bU\.?D\.?R\.?\b", "UDR"),
    (r"\bF\.?N\.?\b", "FN"),
    (r"\bM\.?R\.?G\.?\b", "MRG"),
    (r"\bC\.?N\.?I\.?\b", "CNI"),
    (r"\bU\.?R\.?C\.?\b", "URC"),
    (r"\bR\.?P\.?C\.?R\.?\b", "RPCR"),
]


def extract_party(text):
    """Extract party label from profession de foi text."""
    text_upper = text.upper()

    # Search full patterns in entire text
    for pattern, label in PARTY_PATTERNS:
        if re.search(pattern, text_upper):
            return label

    # Search abbreviations only in first 1000 characters (header area)
    header = text_upper[:1000]
    for pattern, label in ABBREV_PATTERNS:
        if re.search(pattern, header):
            return label

    return None


df["party_label"] = df["text"].apply(extract_party)

# --- Step 2: Remove party mentions from text to avoid data leakage ---
ALL_LEAK_PATTERNS = [p for p, _ in PARTY_PATTERNS] + [p for p, _ in ABBREV_PATTERNS]


def clean_text(text):
    """Remove party name mentions from text to prevent leakage."""
    cleaned = text
    for pattern in ALL_LEAK_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Collapse multiple spaces/newlines left by removals
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


df["text_clean"] = df["text"].apply(clean_text)

# --- Step 3: Group labels into broader political families ---
LABEL_TO_FAMILY = {
    "LO": "Extreme gauche",
    "LCR": "Extreme gauche",
    "PCF": "Gauche",
    "PS": "Gauche",
    "PSU": "Gauche",
    "MRG": "Gauche",
    "Union Gauche": "Gauche",
    "Verts": "Ecologistes",
    "UDF": "Droite",
    "RI": "Droite",
    "CNI": "Droite",
    "RPR": "Droite",
    "UDR": "Droite",
    "URC": "Droite",
    "UPF": "Droite",
    "Maj. Pres.": "Droite",
    "FN": "Extreme droite",
    "PFN": "Extreme droite",
    "RPCR": "Droite",
}

df["party_family"] = df["party_label"].map(LABEL_TO_FAMILY)

# --- Step 4: Drop unlabeled and Ecologistes (too few samples: 38) ---
df = df.dropna(subset=["party_label"])
df = df[df["party_family"] != "Ecologistes"]

# --- Statistics ---
total = len(df)
print(f"Total labeled documents: {total}")
print()
print("Distribution by party:")
print(df["party_label"].value_counts().to_string())
print()
print("Distribution by family:")
print(df["party_family"].value_counts().to_string())
print()
print("Distribution by year:")
print(df.groupby("year")["party_label"].count().to_string())

df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")

# --- Step 5: Train/val/test split (stratified by party_family and year) ---
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42,
    stratify=df["party_family"],
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42,
    stratify=temp_df["party_family"],
)

train_df.to_parquet("data/train.parquet", index=False)
val_df.to_parquet("data/val.parquet", index=False)
test_df.to_parquet("data/test.parquet", index=False)

print(f"\nSplit: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
print(f"Train distribution:\n{train_df['party_family'].value_counts().to_string()}")
