"""
Parse raw text files from the Archelec corpus and aggregate multi-page
documents into a single text per candidate.

Input: text_files/{year}/legislatives/*.txt
Output: data/corpus_by_document.parquet
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

YEARS = [1973, 1978, 1981, 1988, 1993]
OUTPUT_PATH = Path("data/corpus_by_document.parquet")

# Group pages by document key (everything except the page number)
documents = defaultdict(lambda: {"pages": [], "year": "", "dept": "", "constituency": "", "round": ""})

for year in YEARS:
    folder = Path(f"text_files/{year}/legislatives")
    if not folder.exists():
        continue
    for txt_file in folder.glob("*.txt"):
        # Only keep professions de foi, skip ballots (BV) and OCR artifacts
        if "_PF_" not in txt_file.name or "pdfmasterocr" in txt_file.name:
            continue

        text = txt_file.read_text(encoding="utf-8")
        parts = txt_file.stem.split("_")

        # doc_key = everything except page number, used to group pages
        doc_key = "_".join(parts[:8])
        page = parts[8]

        documents[doc_key]["pages"].append((page, text))
        documents[doc_key]["year"] = parts[2]
        documents[doc_key]["dept"] = parts[4]
        documents[doc_key]["constituency"] = parts[5]
        documents[doc_key]["round"] = parts[6]

# Concatenate pages in order and build DataFrame
rows = []
for doc_key, doc in documents.items():
    sorted_pages = sorted(doc["pages"], key=lambda x: x[0])
    full_text = "\n".join(text for _, text in sorted_pages)
    rows.append({
        "doc_name": doc_key,
        "year": doc["year"],
        "dept_code": doc["dept"],
        "constituency": doc["constituency"],
        "round": doc["round"],
        "text": full_text,
    })

df = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)

print(f"Documents: {len(df)}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Saved to {OUTPUT_PATH}")
