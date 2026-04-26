# `results/` — what's where

This directory holds every artefact produced by the pipeline (training
metrics, per-sample predictions, merged tables, error breakdowns,
figures, and case studies) for all three datasets. Files are kept flat
on disk to match the path conventions in the analysis scripts; this
README groups them by dataset for navigation.

The repository's main [`README.md`](../README.md) has the full
methodology and headline numbers; this document is a file index.

---

## SST-2 (sentiment, single-sentence)

**Training metrics**
- [`train_bert_sst2.json`](train_bert_sst2.json) — BERT-base training run
- [`distilbert_sst2_config.json`](distilbert_sst2_config.json) — DistilBERT config + final eval
- [`distilbert_sst2_results.txt`](distilbert_sst2_results.txt) — DistilBERT summary (parsed for inference latency)

**Per-sample predictions**
- [`predict_bert_sst2.csv`](predict_bert_sst2.csv)
- [`predict_distilbert_sst2.csv`](predict_distilbert_sst2.csv)

**Error analysis**
- [`merged_predictions_sst2.csv`](merged_predictions_sst2.csv) — joined + 4-category labelled
- [`error_breakdown_sst2.json`](error_breakdown_sst2.json) — aggregate stats (length buckets, negation split, confidence)

**Figures** — [`figures_sst2/`](figures_sst2/) (7 PNGs: fig1 overall metrics → fig7 confidence calibration)

**Case studies**
- [`case_studies_sst2.md`](case_studies_sst2.md) — hand-written analysis (12 cases, 4 themes)
- [`case_studies_candidates_sst2.csv`](case_studies_candidates_sst2.csv) — all 41 disagreements, ranked

---

## IMDb (sentiment, long-document)

**Training metrics**
- [`train_bert_imdb.json`](train_bert_imdb.json)
- [`distilbert_imdb_config.json`](distilbert_imdb_config.json)
- [`distilbert_imdb_results.txt`](distilbert_imdb_results.txt)

**Per-sample predictions**
- [`predict_bert_imdb.csv`](predict_bert_imdb.csv)
- [`predict_distilbert_imdb.csv`](predict_distilbert_imdb.csv)

**Error analysis**
- [`merged_predictions_imdb.csv`](merged_predictions_imdb.csv)
- [`error_breakdown_imdb.json`](error_breakdown_imdb.json)

**Figures** — [`figures_imdb/`](figures_imdb/) (same 7 figures as SST-2, IMDb data)

**Case studies**
- [`case_studies_imdb.md`](case_studies_imdb.md) — hand-written analysis (9 cases, 4 themes)
- [`case_studies_candidates_imdb.csv`](case_studies_candidates_imdb.csv) — all 1,190 disagreements, ranked

---

## MRPC (paraphrase identification, sentence pairs)

**Training metrics**
- [`train_bert_mrpc.json`](train_bert_mrpc.json)
- [`distilbert_mrpc_config.json`](distilbert_mrpc_config.json)
- [`distilbert_mrpc_results.txt`](distilbert_mrpc_results.txt)

**Per-sample predictions**
- [`predict_bert_mrpc.csv`](predict_bert_mrpc.csv)
- [`predict_distilbert_mrpc.csv`](predict_distilbert_mrpc.csv)

**Error analysis**
- [`merged_predictions_mrpc.csv`](merged_predictions_mrpc.csv)
- [`error_breakdown_mrpc.json`](error_breakdown_mrpc.json)

**Figures** — [`figures_mrpc/`](figures_mrpc/) (same 7 figures as SST-2, MRPC data)

**Case studies**
- [`case_studies_mrpc.md`](case_studies_mrpc.md) — hand-written analysis (5 cases, 2 themes per direction)
- [`case_studies_candidates_mrpc.csv`](case_studies_candidates_mrpc.csv) — all 56 disagreements, ranked

---

## File-name conventions

The naming scheme encodes the dataset in the filename rather than nesting
directories, so the analysis scripts can stay path-simple:

- `train_bert_{dataset}.json` / `distilbert_{dataset}_config.json` — training metrics
- `predict_{model}_{dataset}.csv` — per-sample predictions
- `merged_predictions_{dataset}.csv` — joined BERT + DistilBERT, labelled by 4-category
- `error_breakdown_{dataset}.json` — aggregate analysis
- `case_studies_{dataset}.md` — hand-written qualitative analysis
- `case_studies_candidates_{dataset}.csv` — ranked disagreements (input to the markdown)
- `figures_{dataset}/` — seven PNGs per dataset

Model checkpoint directories (`{dataset}_bert_base/` and
`distilbert_{dataset}_model/`) are written by the training scripts but
not committed (≥270 MB each).
