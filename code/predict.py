"""
Generate per-sample prediction CSVs from a fine-tuned model checkpoint.

Reads:
    results/sst2_bert_base/             (output of train_bert.py)
    results/distilbert_sst2_model/      (output of train_distilbert.py)

Writes:
    results/predict_bert_sst2.csv
    results/predict_distilbert_sst2.csv

Each CSV has one row per SST-2 validation sample with:
    id, text, true_label,
    {model}_pred, {model}_confidence, {model}_prob_label_1, {model}_correct,
    text_length_words, text_length_chars, has_negation

Run from the repo root:
    python code/predict.py --model bert
    python code/predict.py --model distilbert

Or both at once:
    python code/predict.py --model both
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)


NEG_WORDS = ["not", "no", "never", "n't", "but", "however", "although"]

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MODEL_CONFIG = {
    "bert": {
        "model_dir": RESULTS / "sst2_bert_base",
        "output_csv": RESULTS / "predict_bert_sst2.csv",
        "label": "bert",
    },
    "distilbert": {
        "model_dir": RESULTS / "distilbert_sst2_model",
        "output_csv": RESULTS / "predict_distilbert_sst2.csv",
        "label": "distilbert",
    },
}


def has_negation(text):
    text = text.lower()
    return int(any(w in text for w in NEG_WORDS))


def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def make_predictions(model_dir, output_csv, model_label, max_length=128):
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run the corresponding training script first "
            f"(train_bert.py or train_distilbert.py)."
        )

    raw = load_dataset("glue", "sst2")
    val_texts = raw["validation"]["sentence"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(ex):
        return tokenizer(ex["sentence"], truncation=True, max_length=max_length)

    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized.set_format("torch")

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    pred_out = trainer.predict(tokenized["validation"])

    logits = pred_out.predictions
    labels = pred_out.label_ids
    probs = softmax_np(logits)
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    prob_label_1 = probs[:, 1]

    df = pd.DataFrame({
        "id": range(len(val_texts)),
        "text": val_texts,
        "true_label": labels.astype(int),
        f"{model_label}_pred": pred_labels.astype(int),
        f"{model_label}_confidence": confidences.astype(float),
        f"{model_label}_prob_label_1": prob_label_1.astype(float),
        f"{model_label}_correct": (pred_labels == labels).astype(int),
        "text_length_words": [len(str(t).split()) for t in val_texts],
        "text_length_chars": [len(str(t)) for t in val_texts],
        "has_negation": [has_negation(t) for t in val_texts],
    })

    os.makedirs(Path(output_csv).parent, exist_ok=True)
    df.to_csv(output_csv, index=False)

    accuracy = df[f"{model_label}_correct"].mean()
    print(f"  saved {len(df)} rows to {Path(output_csv).relative_to(ROOT)}")
    print(f"  accuracy ({model_label}): {accuracy:.4f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "distilbert", "both"],
                        default="both")
    args = parser.parse_args()

    targets = ["bert", "distilbert"] if args.model == "both" else [args.model]
    for name in targets:
        cfg = MODEL_CONFIG[name]
        print(f"\n[{name}]")
        make_predictions(cfg["model_dir"], cfg["output_csv"], cfg["label"])


if __name__ == "__main__":
    main()
