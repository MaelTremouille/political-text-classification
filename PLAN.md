# Plan: Discourse versus Economic Reality

## Context

ENSAE 3A NLP project — two students, same corpus (~21,700 OCR'd French legislative campaign manifestos, 1973-1993), two independent research questions. Deadline: **April 30, 2026**.

Each student writes a standalone NeurIPS-format report covering: problem definition, data description, state of the art, model justification, experiments, and results analysis.

---

## Step 0 — Shared: Data Preparation (`src/data_preparation.py`)

**Owner**: shared (whoever starts first)

**Goal**: Parse filenames, extract metadata, aggregate text at document level.

- Parse filename fields: `{election_id}_{type}_{year}_{month}_{dept}_{constituency}_{round}_{doc_type}_{page}`
- Department code = `parts[4]` (3 chars)
- Keep only `_PF_` files, discard `_BV_` and `_pdfmasterocr`
- Concatenate pages of the same candidate's profession de foi into a single document
- Output: `data/corpus_by_document.parquet` with columns `(year, dept_code, constituency, round, doc_name, text)`

**Status**: [ ] Not started

---

## Part A — Topic Modeling: Discourse vs Economic Reality

**Owner**: Giovanni (I suggest you edit what you want to do but here is an example of what can be done)

**Research question**: Do candidates talk more about unemployment/agriculture in departments where these economic issues are actually more pressing?

### Step A1 — BERTopic Pipeline (`src/topic_analysis.py`)

- Embed documents with CamemBERT (`sentence-transformers`)
- UMAP dimensionality reduction + HDBSCAN clustering
- c-TF-IDF for topic labeling
- Output: topic assignments per document, topic distribution per (year, dept)
- Save: `data/topic_distributions.parquet`, `data/document_embeddings.npy`

**Status**: [ ] Not started

### Step A2 — Economic Data (`src/economic_data.py`)

- Collect department-level indicators from INSEE (unemployment rate, agricultural/industrial employment share)
- Clean and harmonize into `data/economic_indicators.parquet`
- Handle Corsica codes (20 vs 2A/2B)

**Status**: [ ] Not started

### Step A3 — Embeddings Analysis (`src/embeddings_analysis.py`)

- Mean embeddings per department-year and per party-year
- t-SNE / UMAP 2D visualizations colored by party, year, unemployment rate
- Cosine distance analysis (inter/intra group comparisons)

**Status**: [ ] Not started

### Step A4 — Cross-Analysis (`src/analysis.py`)

- Merge topic distributions + economic indicators on (year, dept_code)
- Spearman correlations between topic shares and economic indicators
- OLS regressions with year fixed effects
- Scatter plots with regression lines
- Output: `data/merged_dataset.parquet`, regression tables, figures

**Status**: [ ] Not started

---

## Part B — Political Affiliation Classification

**Owner**: Mael

**Research question**: Can we predict a candidate's political affiliation from their campaign text alone? What semantic proximities exist between parties?

### Step B1 — Label Extraction & Dataset Construction

- Extract party labels from text via regex (RPR, PS, PCF, UDF, FN, Verts, MRG, etc.)
- Build labeled dataset: `(doc_name, year, dept, text, party_label)`
- Descriptive statistics: label distribution by year, class balance analysis
- Train/validation/test split (stratified by party and year)

**Status**: [ ] Not started

### Step B2 — Text Classification (`src/classification.py`)

- Baseline: TF-IDF + Logistic Regression / SVM
- Main model: fine-tuned CamemBERT for multi-class classification
- Evaluation: accuracy, F1 (macro/weighted), confusion matrices
- Analysis: which parties are most confused? Does this reflect real political proximity?

**Status**: [ ] Not started

### Step B3 — Semantic Mapping (`src/political_mapping.py`)

- CamemBERT embeddings (from classification or sentence-transformers)
- Mean embeddings per party per year
- t-SNE / UMAP 2D projections colored by party
- Cosine similarity heatmaps between parties
- Temporal evolution: how do party positions shift over 1973-1993?

**Status**: [ ] Not started

### Step B4 — Results & Visualization

- Confusion matrix analysis → political proximity interpretation
- 2D maps with party trajectories over time
- Compare classifier performance across years (does discourse become more/less distinguishable?)
- Figures for the report

**Status**: [ ] Not started

---

## Interface Between Parts

Shared key: `(year, dept_code)` and `(doc_name)`

| Part A produces | Part B produces |
|-----------------|-----------------|
| Topic per document | Party label per document |
| Topic distribution per (year, dept) | Party distribution per (year, dept) |
| Document embeddings | Classification embeddings |

**Optional cross-analysis**: Is the dominant topic of a party consistent with its political identity? Do parties in high-unemployment departments emphasize different topics?

---

## Timeline

| Week | Mael (Part B) | Camarade (Part A) |
|------|---------------|-------------------|
| W1 | Step 0 (shared data prep) + Step B1 (label extraction) | Step A2 (INSEE data) |
| W2 | Step B2 (classification pipeline) | Step A1 (BERTopic) |
| W3 | Step B3 (semantic mapping) | Step A3 (embeddings) + Step A4 (cross-analysis) |
| W4 | Step B4 (results, figures, report) | Results, figures, report |

---

## Dependencies

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
transformers>=4.30
sentence-transformers>=2.2
bertopic>=0.16          # Part A
umap-learn>=0.5
hdbscan>=0.8            # Part A
statsmodels>=0.13       # Part A
unidecode>=1.3
jupyter>=1.0
torch>=2.0
```
