# Political affiliation classification from French legislative campaign manifestos (1973–1993)

NLP project for ENSAE Paris (2025–2026). We classify political affiliation from *professions de foi* using TF-IDF and CamemBERT, and analyze inter-party semantic proximity with sentence-transformer embeddings.

## Data

The corpus comes from the [Archelec project](https://archelec.sciencespo.fr) (Sciences Po). The OCR'd text files were obtained from the course repository:

```
git clone https://gitlab.teklia.com/ckermorvant/arkindex_archelec.git
```

Raw texts are in `text_files/` as zip archives. Run `python extract_text.py` to unzip them.

## Pipeline

```
python src/data_preparation.py      # parse and aggregate multi-page documents
python src/label_extraction.py      # extract party labels, clean text, split data
python src/classification.py        # train TF-IDF baseline + fine-tune CamemBERT
python src/political_mapping.py     # compute embeddings, t-SNE, cosine similarity
```

## Main results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TF-IDF + Logistic Regression | **0.84** | **0.80** | **0.85** |
| CamemBERT (best run) | 0.71 | 0.65 | 0.72 |

TF-IDF outperforms CamemBERT on every class. We attribute this to the small dataset size (3,163 training examples) and the fact that the signal is mostly in specific keywords rather than context. Extrême gauche is the easiest family to classify (F1 = 0.96), while Droite is the hardest (F1 = 0.60) due to overlap with Gauche vocabulary.

For semantic mapping, sentence-transformer embeddings show that most parties have cosine similarity above 0.90. The RPCR (a regionalist party from New Caledonia) is the main outlier (0.70–0.78).

## Report

The report is in `report/main.tex` (NeurIPS format). Figures are in `figures/`.
