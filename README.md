# DistilBERT vs BERT — error analysis on SST-2 and IMDb

We fine-tune `bert-base-uncased` and `distilbert-base-uncased` on two
sentiment-classification benchmarks — **SST-2** (short, single-sentence)
and **IMDb** (long, multi-paragraph reviews) — and study **what kinds of
inputs the distilled student loses on relative to the teacher**.

> Research question: *How much performance is lost by distillation, and
> what kinds of examples are most affected?*

---

## Headline results

### Aggregate metrics

|  | SST-2 (n = 872 val) |  | IMDb (n = 25,000 test) |  |
|---|---:|---:|---:|---:|
|  | **BERT** | **DistilBERT** | **BERT** | **DistilBERT** |
| Accuracy | **0.928** | 0.899 | **0.924** | 0.916 |
| F1 | **0.930** | 0.902 | **0.925** | 0.916 |
| Accuracy gap (pp) |  | −2.87 |  | −0.84 |
| Training time (3 epochs, T4) | 391 s | 275 s | 1013 s | 530 s |
| Training speed-up (DistilBERT vs BERT) |  | 1.42× |  | 1.91× |
| Parameters | 109.5 M | 67.0 M | 109.5 M | 67.0 M |
| Inference latency (ms / sample, eval throughput) | 2.24 | 1.98 | 3.48 | 1.87 |
| Inference speed-up (DistilBERT vs BERT) |  | 1.13× |  | 1.86× |
| Agreement rate (the two models give the same prediction) |  | 95.3% |  | 95.2% |

> Latency is computed as `1 / eval_samples_per_second` from `trainer.evaluate()` on the full evaluation set, with batch size 16 — same pipeline for both models. We do not use a separate warm-up benchmark; with 25K samples on IMDb (872 on SST-2) any warm-up overhead is dominated by averaging.

DistilBERT is consistently the smaller, faster, slightly less accurate
model. The headline accuracy gap is much larger on SST-2 (−2.87 pp)
than on IMDb (−0.84 pp), but **the per-bucket story flips with input
length** — see below.

### Comparison to the DistilBERT paper

The original paper (Sanh et al., 2019) reports DistilBERT as 40% smaller,
60% faster, and retaining ~97% of BERT's language-understanding ability.
Our numbers track those claims:

| Paper claim | Our measurement | Trend |
|---|---|:-:|
| 40% smaller | 39% smaller (67.0M / 109.5M params) | ✓ |
| 60% faster (training) | 30% (SST-2) / 48% (IMDb) faster training | partial |
| Retains ~97% of capability | 96.9% (SST-2) / 99.1% (IMDb) of BERT's accuracy | ✓ |

The size and accuracy-retention numbers replicate the paper's headline
trends almost exactly. Our training-speed advantage is smaller than the
paper's claimed 60% — likely because both models hit the same input-pipe
and tokenization overhead on a single T4, which dominates wall-clock at
our batch sizes more than at the paper's training scale.

### Where the gap lives

The two datasets stress the student in different ways:

|  | SST-2 (short, single-sentence) | IMDb (long, multi-paragraph) |
|---|---|---|
| Headline gap | **−2.87 pp** | **−0.84 pp** |
| Length effect | 1.7 pp (5–10 words) → 3.1 pp (20+ words) | 0.3 pp (<100 w) → **1.9 pp (400+ words)** |
| Negation effect | gap doubles on negated sentences (4.3 pp vs 2.0 pp) | 97% of reviews contain a negation; flag is uninformative |
| Dominant failure mode | local compositionality (negation, sarcasm, idiom) | long-document integration (buried verdicts, mixed sentiment) |
| DistilBERT confidence on its mistakes | 0.893 (n = 33) | 0.867 (n = 700) |

The unifying observation: **DistilBERT is most likely to fail when
sentiment is encoded compositionally rather than additively** — through
negation/sarcasm in short text (SST-2), or through where in a long review
the verdict lives (IMDb). And in both cases the student is *overconfident*
on its mistakes, so simple confidence-thresholding will not catch them.

Full numbers:
[`results/error_breakdown.json`](results/error_breakdown.json) (SST-2) ·
[`results/error_breakdown_imdb.json`](results/error_breakdown_imdb.json) (IMDb).
Figures:
[`results/figures/`](results/figures/) ·
[`results/figures_imdb/`](results/figures_imdb/).
Hand-annotated qualitative analysis:
[`results/case_studies.md`](results/case_studies.md) ·
[`results/case_studies_imdb.md`](results/case_studies_imdb.md).

---

## Repository structure

```
DistilBERT/
├── code/
│   ├── train_bert.py              # fine-tune bert-base-uncased (SST-2 or IMDb via --dataset)
│   ├── train_distilbert.py        # fine-tune distilbert-base-uncased on SST-2
│   ├── train_bert_imdb.py         # fine-tune bert-base-uncased on IMDb
│   ├── train_distilbert_imdb.py   # fine-tune distilbert-base-uncased on IMDb
│   ├── predict.py                 # load fine-tuned model, write per-sample CSV (--dataset)
│   ├── predict_bert.py            # earlier all-in-one (train+predict) variant
│   ├── predict_distilbert.py      # earlier predict-only variant (Colab paths)
│   ├── error_analysis.py          # 4-category breakdown + length / negation / confidence stats
│   ├── extract_cases.py           # rank disagreements → case_studies_candidates*.csv
│   └── make_figures.py            # 7 figures per dataset
│
├── data/
│   ├── SST2.py                    # GLUE SST-2 loader + tokenizer wrapper
│   └── IMDB.py                    # IMDb loader + tokenizer wrapper
│
├── results/
│   ├── train_bert_sst2.json                  # SST-2 BERT training metrics
│   ├── train_bert_sst2_history.json
│   ├── distilbert_sst2_config.json           # SST-2 DistilBERT training metrics
│   ├── distilbert_sst2_results.txt
│   ├── predict_bert_sst2.csv                 # SST-2 per-sample BERT predictions
│   ├── predict_distilbert_sst2.csv           # SST-2 per-sample DistilBERT predictions
│   ├── merged_predictions_sst2.csv           # joined + 4-category labelled
│   ├── error_breakdown.json                  # SST-2 aggregate error analysis
│   ├── case_studies.md                       # SST-2 qualitative analysis (12 cases, 4 themes)
│   ├── case_studies_candidates.csv           # SST-2 disagreements, ranked
│   ├── figures/                              # SST-2 figures (7 PNGs)
│   │
│   ├── train_bert_imdb.json                  # IMDb BERT training metrics
│   ├── distilbert_imdb_config.json           # IMDb DistilBERT training metrics
│   ├── distilbert_imdb_results.txt
│   ├── predict_bert_imdb.csv                 # IMDb per-sample BERT predictions
│   ├── predict_distilbert_imdb.csv           # IMDb per-sample DistilBERT predictions
│   ├── merged_predictions_imdb.csv           # joined + 4-category labelled
│   ├── error_breakdown_imdb.json             # IMDb aggregate error analysis
│   ├── case_studies_imdb.md                  # IMDb qualitative analysis (9 cases, 4 themes)
│   ├── case_studies_candidates_imdb.csv      # IMDb disagreements, ranked
│   └── figures_imdb/                         # IMDb figures (7 PNGs)
│
└── README.md
```

---

## Reproducing the experiments

The training steps need a GPU; the analysis steps run locally on CPU.

### 1. Setup

```bash
git clone https://github.com/JodieeT/DistilBERT.git
cd DistilBERT
pip install transformers datasets evaluate scikit-learn pandas matplotlib
```

### 2. Fine-tune both models

```bash
# SST-2  (~12 min on a T4)
python code/train_bert.py --dataset sst2 --batch_size 32 --epochs 3
python code/train_distilbert.py

# IMDb  (~26 min on a T4)
python code/train_bert_imdb.py
python code/train_distilbert_imdb.py
```

Writes the fine-tuned model checkpoints to `results/{sst2,imdb}_bert_base/`
and `results/distilbert_{sst2,imdb}_model/`. Model weight files are large
(~440 MB BERT, ~270 MB DistilBERT) and are *not* committed — only the small
metric JSON/TXT files are.

### 3. Generate per-sample prediction CSVs

```bash
python code/predict.py --model both --dataset sst2
python code/predict.py --model both --dataset imdb
```

Writes `results/predict_{bert,distilbert}_{sst2,imdb}.csv` — one row per
evaluation sample, with text, true label, predicted label, confidence,
length, and a negation flag.

(`predict_bert.py` and `predict_distilbert.py` are earlier variants kept
for record; `predict_distilbert.py` has hardcoded Colab paths and will not
run as-is locally.)

### 4. Run the error analysis

```bash
python code/error_analysis.py --dataset sst2
python code/error_analysis.py --dataset imdb
```

Joins the two prediction CSVs for each dataset, labels each row with one
of four categories (`both_correct`, `both_wrong`, `bert_only_correct`,
`distilbert_only_correct`), and computes per-category statistics, length
buckets, negation splits, and confidence stats. Outputs:

- `results/merged_predictions_{dataset}.csv`
- `results/error_breakdown.json` (SST-2) / `results/error_breakdown_imdb.json` (IMDb)

### 5. Generate figures

```bash
python code/make_figures.py --dataset sst2
python code/make_figures.py --dataset imdb
```

Writes seven figures per dataset, to `results/figures/` and
`results/figures_imdb/` respectively:

| File | What it shows |
|---|---|
| `fig1_overall_metrics.png` | accuracy, F1, parameter count, training time, inference latency |
| `fig2_error_categories.png` | how the samples split across the 4 categories |
| `fig3_length_buckets.png` | accuracy vs text length (per-dataset bucketing) |
| `fig4_negation_split.png` | accuracy on sentences with vs without negation/contrast markers |
| `fig5_length_per_category.png` | text-length distribution per error category |
| `fig6_confidence_scatter.png` | per-sample BERT vs DistilBERT confidence, coloured by category |
| `fig7_confidence_calibration.png` | mean confidence on correct vs wrong predictions |

### 6. Inspect the disagreement cases

```bash
python code/extract_cases.py --dataset sst2
python code/extract_cases.py --dataset imdb
```

Writes `results/case_studies_candidates{,_imdb}.csv` — all disagreements
between the two models, ranked by how confidently the wrong model held
its position. The hand-curated subsets with linguistic-phenomenon labels
live in [`results/case_studies.md`](results/case_studies.md) (SST-2) and
[`results/case_studies_imdb.md`](results/case_studies_imdb.md) (IMDb).

---

## Findings

### SST-2 (short, single-sentence) — 41 disagreements, 4 themes

1. **Polarity reversal via negation** — a single negation flips an
   otherwise positive surface (*"i don't think i laughed out loud once"*).
2. **Contrastive / concessive structure** — verdict is in the second
   clause after `but / though` (*"outer-space buffs might love this film,
   but others will find its pleasures intermittent"*).
3. **Sarcasm and metaphor** — surface words point the wrong way (*"the
   iditarod lasts for days — this just felt like it did"*).
4. **Understated / idiomatic positive** — *almost unsurpassed*, *not the
   least of which*, *shades of gray* all read as negative on the surface.

Per-case discussion: [`results/case_studies.md`](results/case_studies.md).

### IMDb (long, multi-paragraph) — 1,190 disagreements, 4 themes

1. **Buried verdict in long review** — plot summary or descriptive praise
   dominates the document; the verdict arrives in the last paragraph and
   DistilBERT misses it.
2. **Temporal / sentiment reversal** — *"I loved this movie as a kid…
   watching it again as an adult, this film is terrible."*
3. **Ironic / understated polarity** — ostensibly positive phrasing
   conveys a negative verdict (*"great movie to sit down with a six-pack
   — DO NOT see this movie sober"*).
4. **Reverse direction (DistilBERT correct, BERT wrong)** — most often
   contrastive ("X is a mess, **but** an entertaining mess"), where BERT
   over-weights the negative descriptors before the pivot.

Per-case discussion: [`results/case_studies_imdb.md`](results/case_studies_imdb.md).

### Cross-dataset

DistilBERT is fine on short, sentiment-rich text where bag-of-tokens cues
align with the label, and breaks where sentiment is encoded
*compositionally* — through negation, contrastive structure, sarcasm, or
idiom (SST-2), or through the position of the verdict inside a long
mixed-sentiment document (IMDb). The student's mean confidence on its
mistakes is essentially as high as on its successes (0.87–0.89 in both
datasets), so simple rejection-by-confidence is not a useful safety net.

---

## References

- Devlin, J. *et al.* (2019). *BERT: Pre-training of deep bidirectional
  transformers for language understanding.*
  [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Maas, A. L. *et al.* (2011). *Learning word vectors for sentiment
  analysis.* (IMDb dataset)
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).
  *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and
  lighter.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Socher, R. *et al.* (2013). *Recursive deep models for semantic
  compositionality over a sentiment treebank.* (SST-2 dataset)
