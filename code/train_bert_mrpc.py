"""
Fine-tune bert-base-uncased on GLUE MRPC (paraphrase identification).

Reads the dataset via data/MRPC.py.

Writes:
    results/mrpc_bert_base/         fine-tuned model + tokenizer
    results/train_bert_mrpc.json    metrics in the format expected by make_figures.py

Run from the repo root (needs GPU):
    python code/train_bert_mrpc.py
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import evaluate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data.MRPC import load_mrpc_data


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
OUTPUT_DIR = "results/mrpc_bert_base"
METRICS_PATH = "results/train_bert_mrpc.json"


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        **accuracy.compute(predictions=predictions, references=labels),
        **f1.compute(predictions=predictions, references=labels),
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset, tokenizer, data_collator = load_mrpc_data(
        model_name=MODEL_NAME, max_length=MAX_LENGTH
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    eval_result = trainer.evaluate()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    results = {
        "dataset": "MRPC",
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "weight_decay": 0.01,
        "trainable_parameters": count_parameters(model),
        "training_time_seconds": training_time,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== BERT-base on MRPC =====")
    print(f"Validation Accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"Validation F1:       {eval_result['eval_f1']:.4f}")
    print(f"Training Time:       {training_time:.2f} seconds")
    print(f"Trainable Parameters: {count_parameters(model)}")
    print(f"Metrics saved to:    {METRICS_PATH}")


if __name__ == "__main__":
    main()
