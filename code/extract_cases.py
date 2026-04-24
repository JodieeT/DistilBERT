"""
Extract candidate qualitative case studies from the merged predictions.

Reads:
    results/merged_predictions_sst2.csv

Writes:
    results/case_studies_candidates.csv

The script does NOT pick the "best" cases automatically. It outputs every
sample where the two models disagree, ranked by how confidently the wrong
model held its position. Use this CSV as raw material for the hand-written
analysis in `results/case_studies.md`.

Run from the repo root:
    python code/extract_cases.py
"""

from pathlib import Path

import pandas as pd


CONTRASTIVE_MARKERS = (" but ", " though ", " however ", " although ",
                       " yet ", " still ", " except ")
NEGATION_TOKENS = (" not ", " no ", " never ", "n't ", " neither ", " nor ")


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def count_negations(text):
    t = " " + text.lower() + " "
    return sum(t.count(tok) for tok in NEGATION_TOKENS)


def has_contrastive(text):
    t = " " + text.lower() + " "
    return int(any(m in t for m in CONTRASTIVE_MARKERS))


def main():
    df = pd.read_csv(RESULTS / "merged_predictions_sst2.csv")

    df["n_negations"] = df["text"].apply(count_negations)
    df["has_contrastive"] = df["text"].apply(has_contrastive)

    # The two interesting buckets
    bert_only = df[df.category == "bert_only_correct"].copy()
    distil_only = df[df.category == "distilbert_only_correct"].copy()

    # Rank `bert_only_correct` by how confidently DistilBERT was wrong
    bert_only = bert_only.sort_values("distilbert_confidence", ascending=False)

    # Rank `distilbert_only_correct` by how confidently BERT was wrong
    distil_only = distil_only.sort_values("bert_confidence", ascending=False)

    keep_cols = [
        "id", "category", "text", "true_label",
        "bert_pred", "bert_confidence",
        "distilbert_pred", "distilbert_confidence",
        "text_length_words", "has_negation", "n_negations", "has_contrastive",
    ]

    out = pd.concat([bert_only[keep_cols], distil_only[keep_cols]], axis=0)

    out_path = RESULTS / "case_studies_candidates.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} candidate cases to {out_path.relative_to(ROOT)}")
    print()
    print("=" * 60)
    print(f"BERT-only correct (n={len(bert_only)}) — top 10 by DistilBERT over-confidence")
    print("=" * 60)
    for _, row in bert_only.head(10).iterrows():
        print(f"  [id={row.id:>3}]  true={row.true_label}  distil_pred={row.distilbert_pred}  "
              f"distil_conf={row.distilbert_confidence:.3f}  "
              f"len={row.text_length_words}  neg={row.n_negations}  contrast={row.has_contrastive}")
        print(f"    {row.text.strip()}")
        print()

    print("=" * 60)
    print(f"DistilBERT-only correct (n={len(distil_only)}) — all, by BERT over-confidence")
    print("=" * 60)
    for _, row in distil_only.iterrows():
        print(f"  [id={row.id:>3}]  true={row.true_label}  bert_pred={row.bert_pred}  "
              f"bert_conf={row.bert_confidence:.3f}  "
              f"len={row.text_length_words}  neg={row.n_negations}  contrast={row.has_contrastive}")
        print(f"    {row.text.strip()}")
        print()


if __name__ == "__main__":
    main()
