"""
Generate per-sample prediction CSVs for the fine-tuned MRPC checkpoints.

MRPC is a sentence-pair task (not single-text), so it uses pair tokenization
(`tokenizer(s1, s2)`) instead of the single-text path in `predict.py`.

Reads:
    results/mrpc_bert_base/         fine-tuned BERT
    results/distilbert_mrpc_model/  fine-tuned DistilBERT

Writes:
    results/predict_bert_mrpc.csv
    results/predict_distilbert_mrpc.csv

Each CSV has one row per validation sample with:
    id, sentence1, sentence2, true_label,
    {model}_pred, {model}_confidence, {model}_prob_label_1, {model}_correct,
    text_length_words, text_length_chars, has_negation

Run from the repo root:
    python code/predict_mrpc.py --model both
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
MAX_LENGTH = 128
SPLIT = "validation"

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def model_config(model_label):
    if model_label == "bert":
        model_dir = RESULTS / "mrpc_bert_base"
    elif model_label == "distilbert":
        model_dir = RESULTS / "distilbert_mrpc_model"
    else:
        raise ValueError(model_label)
    return {
        "model_dir": model_dir,
        "output_csv": RESULTS / f"predict_{model_label}_mrpc.csv",
        "label": model_label,
    }


def has_negation_pair(s1, s2):
    t = (str(s1) + " " + str(s2)).lower()
    return int(any(w in t for w in NEG_WORDS))


def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def make_predictions(model_dir, output_csv, model_label):
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run the corresponding training script first "
            f"(train_bert_mrpc.py or train_distilbert_mrpc.py)."
        )

    raw = load_dataset("glue", "mrpc")
    split = raw[SPLIT]
    s1_list = split["sentence1"]
    s2_list = split["sentence2"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(ex):
        return tokenizer(
            ex["sentence1"],
            ex["sentence2"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = raw.map(preprocess, batched=True)
    drop_cols = [c for c in ("sentence1", "sentence2", "idx")
                 if c in tokenized[SPLIT].column_names]
    tokenized = tokenized.remove_columns(drop_cols)
    tokenized.set_format("torch")

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    pred_out = trainer.predict(tokenized[SPLIT])

    logits = pred_out.predictions
    labels = pred_out.label_ids
    probs = softmax_np(logits)
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    prob_label_1 = probs[:, 1]

    df = pd.DataFrame({
        "id": range(len(s1_list)),
        "sentence1": s1_list,
        "sentence2": s2_list,
        "true_label": labels.astype(int),
        f"{model_label}_pred": pred_labels.astype(int),
        f"{model_label}_confidence": confidences.astype(float),
        f"{model_label}_prob_label_1": prob_label_1.astype(float),
        f"{model_label}_correct": (pred_labels == labels).astype(int),
        "text_length_words": [len(str(a).split()) + len(str(b).split())
                              for a, b in zip(s1_list, s2_list)],
        "text_length_chars": [len(str(a)) + len(str(b))
                              for a, b in zip(s1_list, s2_list)],
        "has_negation": [has_negation_pair(a, b)
                         for a, b in zip(s1_list, s2_list)],
    })

    os.makedirs(Path(output_csv).parent, exist_ok=True)
    df.to_csv(output_csv, index=False)

    accuracy = df[f"{model_label}_correct"].mean()
    print(f"  saved {len(df)} rows to {Path(output_csv).relative_to(ROOT)}")
    print(f"  accuracy ({model_label} on mrpc): {accuracy:.4f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "distilbert", "both"],
                        default="both")
    args = parser.parse_args()

    targets = ["bert", "distilbert"] if args.model == "both" else [args.model]
    for name in targets:
        cfg = model_config(name)
        print(f"\n[{name} on mrpc]")
        make_predictions(cfg["model_dir"], cfg["output_csv"], cfg["label"])


if __name__ == "__main__":
    main()
