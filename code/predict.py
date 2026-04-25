"""
Generate per-sample prediction CSVs from a fine-tuned model checkpoint.

Reads (depending on --dataset):
    SST-2:  results/sst2_bert_base/  and  results/distilbert_sst2_model/
    IMDb:   results/imdb_bert_base/  and  results/distilbert_imdb_model/

Writes:
    results/predict_bert_{dataset}.csv
    results/predict_distilbert_{dataset}.csv

Each CSV has one row per validation/test sample with:
    id, text, true_label,
    {model}_pred, {model}_confidence, {model}_prob_label_1, {model}_correct,
    text_length_words, text_length_chars, has_negation

Run from the repo root:
    python code/predict.py --model both                 # SST-2 (default)
    python code/predict.py --model both --dataset imdb
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
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


DATASET_CONFIG = {
    "sst2": {
        "loader_args": ("glue", "sst2"),
        "split": "validation",
        "text_column": "sentence",
        "max_length": 128,
    },
    "imdb": {
        "loader_args": ("imdb",),
        "split": "test",
        "text_column": "text",
        "max_length": 256,
    },
}


def model_config(dataset, model_label):
    if model_label == "bert":
        model_dir = RESULTS / f"{dataset}_bert_base"
    elif model_label == "distilbert":
        model_dir = RESULTS / f"distilbert_{dataset}_model"
    else:
        raise ValueError(model_label)
    return {
        "model_dir": model_dir,
        "output_csv": RESULTS / f"predict_{model_label}_{dataset}.csv",
        "label": model_label,
    }


def has_negation(text):
    text = str(text).lower()
    return int(any(w in text for w in NEG_WORDS))


def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_eval_split(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    raw = load_dataset(*cfg["loader_args"])
    split = raw[cfg["split"]]
    texts = split[cfg["text_column"]]
    return raw, split, texts, cfg


def make_predictions(dataset_name, model_dir, output_csv, model_label):
    cfg = DATASET_CONFIG[dataset_name]
    text_col = cfg["text_column"]
    max_length = cfg["max_length"]

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run the corresponding training script first "
            f"(train_bert.py or train_distilbert.py with --dataset {dataset_name})."
        )

    raw, split, val_texts, _ = load_eval_split(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(ex):
        return tokenizer(ex[text_col], truncation=True, max_length=max_length)

    tokenized = raw.map(preprocess, batched=True)
    drop_cols = [c for c in (text_col, "idx") if c in tokenized[cfg["split"]].column_names]
    tokenized = tokenized.remove_columns(drop_cols)
    tokenized.set_format("torch")

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    pred_out = trainer.predict(tokenized[cfg["split"]])

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
    print(f"  accuracy ({model_label} on {dataset_name}): {accuracy:.4f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "distilbert", "both"],
                        default="both")
    parser.add_argument("--dataset", choices=["sst2", "imdb"], default="sst2")
    args = parser.parse_args()

    targets = ["bert", "distilbert"] if args.model == "both" else [args.model]
    for name in targets:
        cfg = model_config(args.dataset, name)
        print(f"\n[{name} on {args.dataset}]")
        make_predictions(args.dataset, cfg["model_dir"], cfg["output_csv"], cfg["label"])


if __name__ == "__main__":
    main()
