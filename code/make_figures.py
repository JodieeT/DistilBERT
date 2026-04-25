"""
Generate figures for the BERT vs DistilBERT error analysis.

Reads (depending on --dataset):
    results/merged_predictions_{dataset}.csv     (output of error_analysis.py)
    results/error_breakdown[_imdb].json          (output of error_analysis.py)
    results/train_bert_{dataset}.json            (output of train_bert.py)
    results/distilbert_{dataset}_config.json     (output of train_distilbert.py)
    results/distilbert_{dataset}_results.txt     (output of train_distilbert.py)

Writes:
    SST-2:  results/figures/figN_*.png
    IMDb:   results/figures_imdb/figN_*.png

Run from the repo root:
    python code/make_figures.py                    # SST-2 (default)
    python code/make_figures.py --dataset imdb
    python code/make_figures.py --dataset mrpc
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


BERT_COLOR = "#3b82f6"      # blue
DISTIL_COLOR = "#f97316"    # orange
GREEN = "#10b981"
RED = "#dc2626"

CAT_COLORS = {
    "both_correct": GREEN,
    "both_wrong": RED,
    "bert_only_correct": BERT_COLOR,
    "distilbert_only_correct": DISTIL_COLOR,
}
CAT_LABELS = {
    "both_correct": "Both correct",
    "both_wrong": "Both wrong",
    "bert_only_correct": "BERT only correct",
    "distilbert_only_correct": "DistilBERT only correct",
}

DATASET_DISPLAY = {"sst2": "SST-2", "imdb": "IMDb", "mrpc": "MRPC"}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_eval_samples_per_second_from_txt(txt_path):
    if not txt_path.exists():
        return None
    text = txt_path.read_text()
    m = re.search(r"'eval_samples_per_second':\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def paths_for(dataset):
    if dataset == "sst2":
        breakdown_path = RESULTS / "error_breakdown.json"
        fig_dir = RESULTS / "figures"
    else:
        breakdown_path = RESULTS / f"error_breakdown_{dataset}.json"
        fig_dir = RESULTS / f"figures_{dataset}"
    return {
        "merged_csv": RESULTS / f"merged_predictions_{dataset}.csv",
        "breakdown": breakdown_path,
        "bert_meta": RESULTS / f"train_bert_{dataset}.json",
        "distil_meta": RESULTS / f"distilbert_{dataset}_config.json",
        "distil_txt": RESULTS / f"distilbert_{dataset}_results.txt",
        "fig_dir": fig_dir,
    }


def load_data(dataset):
    p = paths_for(dataset)
    df = pd.read_csv(p["merged_csv"])
    with open(p["breakdown"]) as f:
        breakdown = json.load(f)
    bert_meta = json.load(open(p["bert_meta"]))
    distil_meta = json.load(open(p["distil_meta"]))
    distil_eval_sps = load_eval_samples_per_second_from_txt(p["distil_txt"])
    return df, breakdown, bert_meta, distil_meta, distil_eval_sps, p["fig_dir"]


def fig1_overall_metrics(breakdown, bert_meta, distil_meta, distil_eval_sps, fig_dir, ds_label):
    overall = breakdown["overall"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

    metrics = ["accuracy", "F1"]
    bert_vals = [overall["bert_accuracy"], overall["bert_f1"]]
    distil_vals = [overall["distilbert_accuracy"], overall["distilbert_f1"]]

    x = np.arange(len(metrics))
    w = 0.36
    axes[0].bar(x - w / 2, bert_vals, w, label="BERT", color=BERT_COLOR)
    axes[0].bar(x + w / 2, distil_vals, w, label="DistilBERT", color=DISTIL_COLOR)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    y_min = min(min(bert_vals), min(distil_vals)) - 0.05
    axes[0].set_ylim(max(0, y_min), 1.0)
    axes[0].set_title(f"Quality (gap: -{overall['accuracy_gap']*100:.2f}pp acc)")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower right", fontsize=9)
    for i, (b, d) in enumerate(zip(bert_vals, distil_vals)):
        axes[0].text(i - w / 2, b + 0.003, f"{b:.3f}", ha="center", fontsize=9)
        axes[0].text(i + w / 2, d + 0.003, f"{d:.3f}", ha="center", fontsize=9)

    bert_params = bert_meta["trainable_parameters"] / 1e6
    distil_params = distil_meta["model_stats"]["num_parameters"] / 1e6
    axes[1].bar(["BERT", "DistilBERT"], [bert_params, distil_params],
                color=[BERT_COLOR, DISTIL_COLOR])
    axes[1].set_ylabel("Million parameters")
    axes[1].set_title(
        f"Parameters\n(DistilBERT = {distil_params / bert_params:.0%} of BERT)"
    )
    for i, v in enumerate([bert_params, distil_params]):
        axes[1].text(i, v + 1.5, f"{v:.1f}M", ha="center", fontsize=10)
    axes[1].set_ylim(0, max(bert_params, distil_params) * 1.15)

    bert_time = bert_meta["training_time_seconds"]
    distil_time = distil_meta["performance"]["training_time_seconds"]
    axes[2].bar(["BERT", "DistilBERT"], [bert_time, distil_time],
                color=[BERT_COLOR, DISTIL_COLOR])
    axes[2].set_ylabel("Seconds")
    axes[2].set_title(
        f"Training time\n(DistilBERT = {distil_time / bert_time:.0%} of BERT)"
    )
    for i, v in enumerate([bert_time, distil_time]):
        axes[2].text(i, v + max(bert_time, distil_time) * 0.02, f"{v:.0f}s",
                     ha="center", fontsize=10)
    axes[2].set_ylim(0, max(bert_time, distil_time) * 1.18)

    bert_eval_sps = bert_meta["eval_metrics"]["eval_samples_per_second"]
    bert_lat_ms = 1000.0 / bert_eval_sps
    if distil_eval_sps is not None:
        distil_lat_ms = 1000.0 / distil_eval_sps
        speedup_str = f"{bert_lat_ms / distil_lat_ms:.2f}x faster"
    else:
        distil_lat_ms = (
            distil_meta["performance"]["avg_inference_time_per_sample_seconds"] * 1000
        )
        speedup_str = "(100-sample est.)"

    axes[3].bar(["BERT", "DistilBERT"], [bert_lat_ms, distil_lat_ms],
                color=[BERT_COLOR, DISTIL_COLOR])
    axes[3].set_ylabel("ms / sample")
    axes[3].set_title(f"Inference latency\n(DistilBERT {speedup_str})")
    for i, v in enumerate([bert_lat_ms, distil_lat_ms]):
        axes[3].text(i, v + max(bert_lat_ms, distil_lat_ms) * 0.02,
                     f"{v:.2f} ms", ha="center", fontsize=10)
    axes[3].set_ylim(0, max(bert_lat_ms, distil_lat_ms) * 1.18)

    fig.suptitle(f"BERT vs DistilBERT — overall comparison on {ds_label}",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    out = fig_dir / "fig1_overall_metrics.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.relative_to(ROOT)}")


def fig2_error_categories(breakdown, fig_dir, ds_label):
    cats = list(CAT_LABELS.keys())
    counts = [breakdown["by_category"][c]["count"] for c in cats]
    pcts = [breakdown["by_category"][c]["pct_of_total"] for c in cats]
    colors = [CAT_COLORS[c] for c in cats]
    labels = [CAT_LABELS[c] for c in cats]
    n_total = breakdown["overall"]["n_samples"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors)
    ax.set_ylabel(f"# samples (of {n_total})")
    ax.set_title(f"Error category breakdown — {ds_label} eval set")
    for bar, p, c in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{c}\n({p}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(counts) * 1.18)
    plt.xticks(rotation=15)
    plt.tight_layout()
    out = fig_dir / "fig2_error_categories.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.relative_to(ROOT)}")


def fig3_length_buckets(breakdown, fig_dir, ds_label):
    rows = breakdown["length_buckets"]
    labels = [r["bucket"] for r in rows]
    bert_acc = [r["bert_acc"] for r in rows]
    distil_acc = [r["distilbert_acc"] for r in rows]
    n = [r["n"] for r in rows]
    gaps = [r["gap"] for r in rows]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, bert_acc, w, label="BERT", color=BERT_COLOR)
    ax.bar(x + w / 2, distil_acc, w, label="DistilBERT", color=DISTIL_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\nn={ni}\ngap={gi:+.3f}"
         for lbl, ni, gi in zip(labels, n, gaps)]
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Text length (words)")
    ax.set_title(f"Accuracy by text length — {ds_label}")
    y_min = min(min(bert_acc), min(distil_acc)) - 0.05
    ax.set_ylim(max(0.0, y_min), 1.05)
    ax.legend(loc="lower right", fontsize=10)
    for i, (b, d) in enumerate(zip(bert_acc, distil_acc)):
        ax.text(i - w / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, d + 0.008, f"{d:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig3_length_buckets.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.relative_to(ROOT)}")


def fig4_negation_split(breakdown, fig_dir, ds_label):
    neg = breakdown["negation_split"]
    labels = ["No negation", "Has negation\n(not / no / never / n't / but / however / although)"]
    bert_acc = [neg["has_negation_0"]["bert_acc"], neg["has_negation_1"]["bert_acc"]]
    distil_acc = [neg["has_negation_0"]["distilbert_acc"],
                  neg["has_negation_1"]["distilbert_acc"]]
    n = [neg["has_negation_0"]["n"], neg["has_negation_1"]["n"]]
    gaps = [neg["has_negation_0"]["gap"], neg["has_negation_1"]["gap"]]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w / 2, bert_acc, w, label="BERT", color=BERT_COLOR)
    ax.bar(x + w / 2, distil_acc, w, label="DistilBERT", color=DISTIL_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\nn={ni}  gap={gi:+.3f}"
         for lbl, ni, gi in zip(labels, n, gaps)]
    )
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy on sentences with vs without negation/contrast markers — {ds_label}")
    y_min = min(min(bert_acc), min(distil_acc)) - 0.05
    ax.set_ylim(max(0.0, y_min), 1.0)
    ax.legend(loc="lower right", fontsize=10)
    for i, (b, d) in enumerate(zip(bert_acc, distil_acc)):
        ax.text(i - w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, d + 0.004, f"{d:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig4_negation_split.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.relative_to(ROOT)}")


def fig5_length_per_category(df, fig_dir, ds_label):
    cats = list(CAT_LABELS.keys())
    data = [df[df.category == c]["text_length_words"].values for c in cats]
    colors = [CAT_COLORS[c] for c in cats]
    labels = [CAT_LABELS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("Text length (words)")
    ax.set_title(f"Text length distribution per error category — {ds_label}")
    plt.xticks(rotation=12)
    plt.tight_layout()
    out = fig_dir / "fig5_length_per_category.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.relative_to(ROOT)}")


def fig6_confidence_scatter(df, fig_dir, ds_label):
    fig, ax = plt.subplots(figsize=(7, 6))
    for cat in CAT_LABELS:
        sub = df[df.category == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub.bert_confidence, sub.distilbert_confidence,
                   alpha=0.45, s=18, label=f"{CAT_LABELS[cat]} (n={len(sub)})",
                   color=CAT_COLORS[cat], edgecolor="none")
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.4, lw=1, label="y = x")
    ax.set_xlabel("BERT confidence")
    ax.set_ylabel("DistilBERT confidence")
    ax.set_title(f"Per-sample confidence — BERT vs DistilBERT ({ds_label})")
    ax.set_xlim(0.49, 1.005)
    ax.set_ylim(0.49, 1.005)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    out = fig_dir / "fig6_confidence_scatter.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.relative_to(ROOT)}")


def fig7_confidence_calibration(df, fig_dir, ds_label):
    models = ["BERT", "DistilBERT"]
    correct = [
        df.loc[df.bert_correct == 1, "bert_confidence"].mean(),
        df.loc[df.distilbert_correct == 1, "distilbert_confidence"].mean(),
    ]
    wrong = [
        df.loc[df.bert_correct == 0, "bert_confidence"].mean(),
        df.loc[df.distilbert_correct == 0, "distilbert_confidence"].mean(),
    ]

    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(x - w / 2, correct, w, label="Correct predictions", color=GREEN)
    ax.bar(x + w / 2, wrong, w, label="Wrong predictions", color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean confidence")
    ax.set_title(f"Calibration — confidence on correct vs wrong predictions ({ds_label})")
    y_min = min(min(correct), min(wrong)) - 0.05
    ax.set_ylim(max(0.0, y_min), 1.0)
    ax.legend(loc="lower right", fontsize=10)
    for i, (cm, wm) in enumerate(zip(correct, wrong)):
        ax.text(i - w / 2, cm + 0.004, f"{cm:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, wm + 0.004, f"{wm:.3f}", ha="center", fontsize=9)

    ax.text(0.5, 0.97,
            f"DistilBERT is more confident when wrong\n(gap to BERT: {wrong[1] - wrong[0]:+.3f})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic", color="#444")
    plt.tight_layout()
    out = fig_dir / "fig7_confidence_calibration.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sst2", "imdb", "mrpc"], default="sst2")
    args = parser.parse_args()
    dataset = args.dataset
    ds_label = DATASET_DISPLAY[dataset]

    df, breakdown, bert_meta, distil_meta, distil_eval_sps, fig_dir = load_data(dataset)
    fig_dir.mkdir(exist_ok=True)

    print(f"Generating figures for {ds_label}...")
    fig1_overall_metrics(breakdown, bert_meta, distil_meta, distil_eval_sps, fig_dir, ds_label)
    fig2_error_categories(breakdown, fig_dir, ds_label)
    fig3_length_buckets(breakdown, fig_dir, ds_label)
    fig4_negation_split(breakdown, fig_dir, ds_label)
    fig5_length_per_category(df, fig_dir, ds_label)
    fig6_confidence_scatter(df, fig_dir, ds_label)
    fig7_confidence_calibration(df, fig_dir, ds_label)
    print(f"\nAll figures saved to {fig_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
