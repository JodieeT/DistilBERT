# DistilBERT vs BERT — error analysis on SST-2

CS 5782 (Cornell Tech) final project. We fine-tune `bert-base-uncased`
and `distilbert-base-uncased` on the SST-2 sentiment classification task
and study **what kinds of sentences the distilled student loses on
relative to the teacher**.

> Research question: *How much performance is lost by distillation, and
> what kinds of examples are most affected?*

---

## Headline results (SST-2 validation, n = 872)

| | BERT-base | DistilBERT-base | Δ |
|---|---:|---:|---:|
| Accuracy | **0.928** | 0.899 | −2.87 pp |
| F1 | **0.930** | 0.902 | −0.028 |
| Parameters | 109.5 M | 67.0 M | 61% of BERT |
| Training time (3 epochs, T4) | 391 s | 275 s | 70% of BERT |
| Inference latency (per sample) | 2.24 ms | 1.98 ms | 1.13× faster |
| Agreement rate | — | — | 95.3% |

The 2.87 pp accuracy gap is concentrated in a few linguistic
phenomena, not spread uniformly:

- **Negation roughly doubles the gap.** Sentences without negation
  markers: gap = 2.0 pp. Sentences with negation
  (`not / no / never / n't / but / however / although`): gap = 4.3 pp.
- **DistilBERT is over-confidently wrong.** Mean confidence on the 33
  cases where BERT is right and DistilBERT is wrong is 0.893 — the
  student does not signal uncertainty when it errs.
- **Length amplifies but does not cause the gap.** 5–10 words: 1.7 pp,
  10–20 words: 2.7 pp, 20+ words: 3.1 pp.

Full numerical breakdown:
[`results/error_breakdown.json`](results/error_breakdown.json).
Figures: [`results/figures/`](results/figures/).
Hand-annotated qualitative analysis:
[`results/case_studies.md`](results/case_studies.md).

---

## Repository structure

```
DistilBERT/
├── code/
│   ├── train_bert.py           # fine-tune bert-base-uncased
│   ├── train_distilbert.py     # fine-tune distilbert-base-uncased
│   ├── predict.py              # load fine-tuned model, write per-sample CSV
│   ├── predict_bert.py         # earlier all-in-one (train+predict) variant
│   ├── predict_distilbert.py   # earlier predict-only variant (Colab paths)
│   ├── error_analysis.py       # 4-category breakdown + length / negation / confidence stats
│   ├── extract_cases.py        # rank disagreements → case_studies_candidates.csv
│   └── make_figures.py         # 7 figures from merged_predictions_sst2.csv
│
├── data/
│   ├── SST2.py                 # GLUE SST-2 loader + tokenizer wrapper
│   └── IMDB.py                 # IMDb loader (used in optional extension)
│
├── results/
│   ├── train_bert_sst2.json              # BERT training metrics
│   ├── distilbert_sst2_config.json       # DistilBERT training metrics
│   ├── distilbert_sst2_results.txt
│   ├── train_bert_sst2_history.json
│   ├── predict_bert_sst2.csv             # per-sample BERT predictions
│   ├── predict_distilbert_sst2.csv       # per-sample DistilBERT predictions
│   ├── merged_predictions_sst2.csv       # joined + 4-category labelled
│   ├── error_breakdown.json              # aggregate error-analysis stats
│   ├── case_studies.md                   # qualitative analysis (12 cases, 4 themes)
│   ├── case_studies_candidates.csv       # all 41 disagreements, ranked
│   └── figures/                          # 7 PNG figures (poster-ready)
│
└── README.md
```

---

## Reproducing the experiments

The training step needs a GPU; the analysis steps run locally on CPU.

### 1. Setup

```bash
git clone https://github.com/JodieeT/DistilBERT.git
cd DistilBERT
pip install transformers datasets evaluate scikit-learn pandas matplotlib
```

### 2. Fine-tune both models on SST-2 (GPU, ~25 min on a T4)

```bash
python code/train_bert.py --dataset sst2 --batch_size 32 --epochs 3
python code/train_distilbert.py
```

This writes the fine-tuned model checkpoints to
`results/sst2_bert_base/` and `results/distilbert_sst2_model/`
respectively. The model weight files are large (~440 MB and ~270 MB)
and are *not* committed to this repo — only the small metric JSON/TXT
files are.

### 3. Generate per-sample prediction CSVs

```bash
python code/predict.py --model both
```

Loads the fine-tuned models from step 2 and writes
`results/predict_bert_sst2.csv` and
`results/predict_distilbert_sst2.csv` — one row per validation sample,
with text, true label, predicted label, confidence, length, and a
negation flag.

(`predict_bert.py` and `predict_distilbert.py` are earlier variants kept
for record; `predict_distilbert.py` has hardcoded Colab paths and will
not run as-is locally.)

### 4. Run the error analysis

```bash
python code/error_analysis.py
```

Joins the two CSVs, labels each row with one of four categories
(`both_correct`, `both_wrong`, `bert_only_correct`,
`distilbert_only_correct`), computes per-category statistics, length
buckets, negation splits, and confidence stats. Outputs:

- `results/merged_predictions_sst2.csv`
- `results/error_breakdown.json`

### 5. Generate figures

```bash
python code/make_figures.py
```

Writes seven PNG figures to `results/figures/`:

| File | What it shows |
|---|---|
| `fig1_overall_metrics.png` | accuracy, F1, parameter count, training time, inference latency |
| `fig2_error_categories.png` | how the 872 samples split across the 4 categories |
| `fig3_length_buckets.png` | accuracy vs text length (5–10 / 10–20 / 20+ words) |
| `fig4_negation_split.png` | accuracy on sentences with vs without negation/contrast markers |
| `fig5_length_per_category.png` | text-length distribution per error category |
| `fig6_confidence_scatter.png` | per-sample BERT vs DistilBERT confidence, coloured by category |
| `fig7_confidence_calibration.png` | mean confidence on correct vs wrong predictions |

### 6. Inspect the disagreement cases

```bash
python code/extract_cases.py
```

Writes `results/case_studies_candidates.csv` — all 41 sentences where
the two models disagree, ranked by how confidently the wrong model
held its position. The hand-curated subset with linguistic-phenomenon
labels lives in [`results/case_studies.md`](results/case_studies.md).

---

## Findings (for the poster)

The qualitative analysis groups the 41 disagreements into four
recurring linguistic patterns:

1. **Polarity reversal via negation** — single negation flips an
   otherwise positive surface (e.g. *"i don't think i laughed out loud
   once"*).
2. **Contrastive / concessive structure** — verdict is in the
   second clause after `but / though` (e.g. *"outer-space buffs might
   love this film, but others will find its pleasures intermittent"*).
3. **Sarcasm and metaphor** — surface words point the wrong way
   (e.g. *"the iditarod lasts for days — this just felt like it did"*).
4. **Understated / idiomatic positive** — phrases like *almost
   unsurpassed*, *not the least of which*, *shades of gray* read as
   negative on the surface.

In all four patterns, sentiment is encoded *compositionally* rather
than through bag-of-positive/negative tokens. The student's reduced
capacity hurts most where local cues *contradict* the true label —
which is also the part of the data where DistilBERT is most
overconfident in its mistakes.

See [`results/case_studies.md`](results/case_studies.md) for the
hand-picked examples and a per-case discussion.

---

## References

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).
  *DistilBERT, a distilled version of BERT: smaller, faster, cheaper
  and lighter.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Devlin, J. *et al.* (2019). *BERT: Pre-training of deep bidirectional
  transformers for language understanding.*
  [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Socher, R. *et al.* (2013). *Recursive deep models for semantic
  compositionality over a sentiment treebank.* (SST-2 dataset)
