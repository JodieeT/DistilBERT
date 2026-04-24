"""
Error analysis: compare BERT and DistilBERT predictions on the same SST-2 validation set.

Reads:
    results/predict_bert_sst2.csv
    results/predict_distilbert_sst2.csv

Writes:
    results/merged_predictions_sst2.csv   per-sample merged table with category column
    results/error_breakdown.json          aggregate stats per category + overall

Run from the repo root:
    python code/error_analysis.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


CATEGORIES = [
    "both_correct",
    "both_wrong",
    "bert_only_correct",
    "distilbert_only_correct",
]


def categorize(row):
    b, d = row["bert_correct"], row["distilbert_correct"]
    if b == 1 and d == 1:
        return "both_correct"
    if b == 0 and d == 0:
        return "both_wrong"
    if b == 1 and d == 0:
        return "bert_only_correct"
    return "distilbert_only_correct"


def category_stats(df):
    out = {}
    n_total = len(df)
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        n = len(sub)
        out[cat] = {
            "count": int(n),
            "pct_of_total": round(100 * n / n_total, 2),
            "mean_text_length_words": round(float(sub["text_length_words"].mean()), 2) if n else None,
            "median_text_length_words": int(sub["text_length_words"].median()) if n else None,
            "mean_text_length_chars": round(float(sub["text_length_chars"].mean()), 2) if n else None,
            "mean_bert_confidence": round(float(sub["bert_confidence"].mean()), 4) if n else None,
            "mean_distilbert_confidence": round(float(sub["distilbert_confidence"].mean()), 4) if n else None,
            "negation_rate": round(float(sub["has_negation"].mean()), 4) if n else None,
            "label_1_rate": round(float((sub["true_label"] == 1).mean()), 4) if n else None,
        }
    return out


def length_bucket_accuracy(df, buckets=((0, 5), (5, 10), (10, 20), (20, 1000))):
    rows = []
    for lo, hi in buckets:
        mask = (df["text_length_words"] >= lo) & (df["text_length_words"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "bucket": f"[{lo},{hi})",
            "n": int(len(sub)),
            "bert_acc": round(float(sub["bert_correct"].mean()), 4),
            "distilbert_acc": round(float(sub["distilbert_correct"].mean()), 4),
            "gap": round(float(sub["bert_correct"].mean() - sub["distilbert_correct"].mean()), 4),
        })
    return rows


def negation_split_accuracy(df):
    out = {}
    for has_neg in (0, 1):
        sub = df[df["has_negation"] == has_neg]
        if len(sub) == 0:
            continue
        out[f"has_negation_{has_neg}"] = {
            "n": int(len(sub)),
            "bert_acc": round(float(sub["bert_correct"].mean()), 4),
            "distilbert_acc": round(float(sub["distilbert_correct"].mean()), 4),
            "gap": round(float(sub["bert_correct"].mean() - sub["distilbert_correct"].mean()), 4),
        }
    return out


def confidence_stats(df):
    return {
        "bert_mean_conf_when_correct": round(float(df.loc[df.bert_correct == 1, "bert_confidence"].mean()), 4),
        "bert_mean_conf_when_wrong": round(float(df.loc[df.bert_correct == 0, "bert_confidence"].mean()), 4),
        "distilbert_mean_conf_when_correct": round(float(df.loc[df.distilbert_correct == 1, "distilbert_confidence"].mean()), 4),
        "distilbert_mean_conf_when_wrong": round(float(df.loc[df.distilbert_correct == 0, "distilbert_confidence"].mean()), 4),
        "mean_abs_confidence_diff": round(float((df["bert_confidence"] - df["distilbert_confidence"]).abs().mean()), 4),
        "pearson_confidence_corr": round(float(df["bert_confidence"].corr(df["distilbert_confidence"])), 4),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"

    bert_csv = results_dir / "predict_bert_sst2.csv"
    distil_csv = results_dir / "predict_distilbert_sst2.csv"

    if not bert_csv.exists() or not distil_csv.exists():
        raise FileNotFoundError(
            f"Expected prediction CSVs at:\n  {bert_csv}\n  {distil_csv}"
        )

    df_b = pd.read_csv(bert_csv)
    df_d = pd.read_csv(distil_csv)

    join_keys = ["id", "text", "true_label", "text_length_words", "text_length_chars", "has_negation"]
    extra_d = ["distilbert_pred", "distilbert_confidence", "distilbert_prob_label_1", "distilbert_correct"]
    df = df_b.merge(df_d[join_keys + extra_d], on=join_keys, how="inner")

    if len(df) != len(df_b):
        raise RuntimeError(f"Merge size mismatch: {len(df)} merged vs {len(df_b)} BERT rows")

    df["category"] = df.apply(categorize, axis=1)

    overall = {
        "n_samples": int(len(df)),
        "bert_accuracy": round(float(df["bert_correct"].mean()), 4),
        "distilbert_accuracy": round(float(df["distilbert_correct"].mean()), 4),
        "accuracy_gap": round(float(df["bert_correct"].mean() - df["distilbert_correct"].mean()), 4),
        "bert_f1": round(float(f1_score(df["true_label"], df["bert_pred"])), 4),
        "distilbert_f1": round(float(f1_score(df["true_label"], df["distilbert_pred"])), 4),
        "agreement_rate": round(float((df["bert_pred"] == df["distilbert_pred"]).mean()), 4),
    }

    breakdown = {
        "overall": overall,
        "by_category": category_stats(df),
        "length_buckets": length_bucket_accuracy(df),
        "negation_split": negation_split_accuracy(df),
        "confidence": confidence_stats(df),
    }

    merged_csv = results_dir / "merged_predictions_sst2.csv"
    df.to_csv(merged_csv, index=False)
    print(f"Merged predictions saved to: {merged_csv}")

    breakdown_path = results_dir / "error_breakdown.json"
    with open(breakdown_path, "w") as f:
        json.dump(breakdown, f, indent=2)
    print(f"Error breakdown saved to:    {breakdown_path}")

    print()
    print("=" * 60)
    print("OVERALL")
    print("=" * 60)
    for k, v in overall.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 60)
    print("BY CATEGORY")
    print("=" * 60)
    for cat in CATEGORIES:
        s = breakdown["by_category"][cat]
        print(f"\n  [{cat}]  n={s['count']} ({s['pct_of_total']}%)")
        print(f"    mean length (words):    {s['mean_text_length_words']}")
        print(f"    mean BERT conf:         {s['mean_bert_confidence']}")
        print(f"    mean DistilBERT conf:   {s['mean_distilbert_confidence']}")
        print(f"    negation rate:          {s['negation_rate']}")

    print()
    print("=" * 60)
    print("LENGTH BUCKETS (bert_acc vs distilbert_acc)")
    print("=" * 60)
    for r in breakdown["length_buckets"]:
        print(f"  {r['bucket']:>10}  n={r['n']:<4}  bert={r['bert_acc']:.4f}  distil={r['distilbert_acc']:.4f}  gap={r['gap']:+.4f}")

    print()
    print("=" * 60)
    print("NEGATION SPLIT")
    print("=" * 60)
    for k, v in breakdown["negation_split"].items():
        print(f"  {k}: n={v['n']}  bert={v['bert_acc']:.4f}  distil={v['distilbert_acc']:.4f}  gap={v['gap']:+.4f}")


if __name__ == "__main__":
    main()
