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
python src/report_analysis.py       # additional stats for the report (text lengths, top features, confusion matrix)
```

## Main results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TF-IDF + Logistic Regression | **0.84** | **0.80** | **0.85** |
| CamemBERT (best run) | 0.71 | 0.65 | 0.72 |

TF-IDF outperforms CamemBERT on every class. The signal is mostly lexical (specific keywords per political family), so a bag-of-words approach captures it well. Extrême gauche is the easiest to classify (F1 = 0.96) thanks to distinctive vocabulary ("travailleurs", class-struggle language), while Droite is the hardest (F1 = 0.60) due to vocabulary overlap with Gauche — the main confusion is 76 Gauche documents predicted as Droite.

Text length varies a lot across families: Extrême gauche averages ~6,700 words vs ~1,100 for Droite, which also contributes to the TF-IDF signal.

For semantic mapping, sentence-transformer embeddings show that most parties have cosine similarity above 0.90. The RPCR (a regionalist party from New Caledonia) is the main outlier (0.70–0.78).

## Report

The report is in `report/main.tex` (NeurIPS format). Figures are in `figures/`.
