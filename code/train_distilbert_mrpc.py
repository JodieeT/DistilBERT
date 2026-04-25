"""
Fine-tune distilbert-base-uncased on GLUE MRPC (paraphrase identification).

Reads the dataset via data/MRPC.py.

Writes:
    results/distilbert_mrpc_model/        fine-tuned model + tokenizer
    results/distilbert_mrpc_results.txt   summary parsed by make_figures.py
    results/distilbert_mrpc_config.json   metrics in the format expected by make_figures.py

Run from the repo root (needs GPU):
    python code/train_distilbert_mrpc.py
"""

import sys
import os
import time
import json
import argparse

import numpy as np
import evaluate
import datasets
import transformers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data.MRPC import load_mrpc_data


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
OUTPUT_DIR = "results/distilbert_mrpc_model"
SUMMARY_PATH = "results/distilbert_mrpc_results.txt"
CONFIG_PATH = "results/distilbert_mrpc_config.json"


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        **accuracy.compute(predictions=preds, references=labels),
        **f1.compute(predictions=preds, references=labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    dataset, tokenizer, data_collator = load_mrpc_data(
        model_name=MODEL_NAME, max_length=MAX_LENGTH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    eval_results = trainer.evaluate()
    print("Evaluation results:")
    print(eval_results)

    sample_dataset = dataset["validation"].select(
        range(min(100, len(dataset["validation"])))
    )
    infer_start = time.time()
    trainer.predict(sample_dataset)
    total_infer_time = time.time() - infer_start
    avg_infer_time = total_infer_time / len(sample_dataset)

    print(f"Total inference time on {len(sample_dataset)} samples: {total_infer_time:.4f} s")
    print(f"Average inference time per sample: {avg_infer_time:.6f} s")

    num_params = model.num_parameters()
    print(f"Number of parameters: {num_params}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

    with open(SUMMARY_PATH, "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write("Dataset: mrpc\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Evaluation results: {eval_results}\n")
        f.write(f"Total inference time on {len(sample_dataset)} samples: {total_infer_time:.4f} seconds\n")
        f.write(f"Average inference time per sample: {avg_infer_time:.6f} seconds\n")
        f.write(f"Number of parameters: {num_params}\n")
    print(f"Summary saved to {SUMMARY_PATH}")

    config = {
        "model": MODEL_NAME,
        "task": "MRPC paraphrase identification",
        "dataset": "mrpc",
        "training": {
            "num_train_epochs": args.epochs,
            "learning_rate": args.lr,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "weight_decay": 0.01,
            "max_length": MAX_LENGTH,
        },
        "evaluation": {
            "metric": "accuracy",
            "final_eval_accuracy": eval_results["eval_accuracy"],
            "final_eval_f1": eval_results.get("eval_f1"),
            "final_eval_loss": eval_results["eval_loss"],
        },
        "performance": {
            "training_time_seconds": training_time,
            "inference_time_100_samples_seconds": total_infer_time,
            "avg_inference_time_per_sample_seconds": avg_infer_time,
        },
        "model_stats": {
            "num_parameters": num_params,
        },
        "environment": {
            "transformers_version": transformers.__version__,
            "datasets_version": datasets.__version__,
            "python_version": sys.version.split()[0],
        },
        "saved_model_path": OUTPUT_DIR,
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
