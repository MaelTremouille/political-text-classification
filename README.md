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

- **TF-IDF + logistic regression**: 84% accuracy (macro F1: 0.80)
- **CamemBERT fine-tuned**: 71% accuracy (macro F1: 0.65)
- TF-IDF outperforms CamemBERT on this corpus — the discriminative signal is mostly lexical
- Sentence-transformer embeddings show high inter-party similarity (>0.90), with the RPCR as the main outlier

## Report

The report is in `report/main.tex` (NeurIPS format). Figures are in `figures/`.
