# Political affiliation classification from French legislative campaign manifestos (1973–1993)

NLP project for ENSAE Paris (2025–2026). We classify political affiliation from *professions de foi* using TF-IDF and CamemBERT, and analyze inter-party semantic proximity with sentence-transformer embeddings.

## Data

The corpus comes from the [Archelec project](https://archelec.sciencespo.fr) (Sciences Po). The OCR'd text files were obtained from the course repository:

```
git clone https://gitlab.teklia.com/ckermorvant/arkindex_archelec.git
```

Raw texts are in `text_files/` as zip archives. Run `python extract_text.py` to unzip them.

## Pipeline

```bash
# 1. Build the corpus and a first cleaned text column
python src/data_preparation.py      # parse and aggregate multi-page documents
python src/label_extraction.py      # extract party labels, clean text, split data

# 2. Train the initial two models (TF-IDF baseline on text_clean, CamemBERT fine-tuning)
python src/classification.py

# 3. Build the aggressively-cleaned column text_clean_v2 (stopwords, dates, toponyms, politician names)
python src/preprocessing.py

# 4. Re-evaluate TF-IDF with 5-fold CV, bootstrap CIs, and temporal splits
python src/evaluation.py --text_col text_clean    --out data/results_tfidf_v1.txt
python src/evaluation.py --text_col text_clean_v2 --out data/results_tfidf_v2.txt

# 5. Data-efficient alternative: frozen sentence-CamemBERT + LogReg
python src/political_mapping.py                   # produces document_embeddings.npy (one-time)
python src/frozen_classifier.py

# 6. Semantic mapping: UMAP, silhouette/Davies–Bouldin, per-party trajectories
python src/semantic_mapping_v2.py

# 7. Error analysis across both classifiers
python src/error_analysis.py
```

## Main results

| Model | Test Acc | Macro F1 | CV Acc (5-fold) | Temporal ≤1981 → ≥1988 |
|-------|----------|----------|------------------|-------------------------|
| TF-IDF + LR (`text_clean`) | 0.841 | 0.795 | 0.835 ± 0.011 | 0.696 |
| TF-IDF + LR (`text_clean_v2`) | **0.842** | 0.794 | **0.850 ± 0.010** | **0.723** |
| CamemBERT fine-tuned (run 2) | 0.710 | 0.650 | – | – |
| sentence-CamemBERT frozen + LR | 0.668 | 0.509 | 0.651 ± 0.010 | 0.422 |

**Summary of findings (full narrative in the report):**

- The baseline TF-IDF's 84% accuracy is partly built on dates, toponyms, and politician surnames, not on political vocabulary. A second cleaning pass (`text_clean_v2`) masks these categories and keeps accuracy essentially unchanged, but the top-20 features for Extrême gauche become unambiguously political (`travailleurs`, `patrons`, `bourgeoisie`, `révolutionnaires`, `payer`).
- Residual leakage on the Droite and Extrême droite classes shifts from first-order (major dates/cities) to second-order (local candidates' surnames and minor cities). This is discussed both as a limitation and as a real feature of French legislative elections, which are candidate-centric.
- A frozen sentence-CamemBERT encoder with a linear head (~4,100 trainable parameters) reaches only 67% test accuracy vs TF-IDF's 84%. Fine-tuning end-to-end (110M parameters on 3,163 training examples) reaches 71% but overfits. On this corpus, sparse lexical features beat contextual embeddings.
- Temporal splits (train on earlier years, test on later ones) drop TF-IDF from 84% to 72% and the frozen head from 67% to 42%. Sentence embeddings are hit harder because topical content shifts more across eras than isolated ideological markers do.
- Semantic mapping via sentence-CamemBERT, t-SNE, and UMAP shows overlapping political families in the embedding space (silhouette ≈ −0.02 on raw 1024-d embeddings, −0.10 on t-SNE, −0.16 on UMAP); Extrême gauche is the only reasonably tight cluster.

## Report

The report (`report/main.tex`, NeurIPS format) compiles to `report/main.pdf`. Figures are shared between `figures/` (generated) and `report/figures/` (used at compile time).

```bash
cd report && pdflatex main.tex && pdflatex main.tex
```
