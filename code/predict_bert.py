import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import argparse
import numpy as np
import pandas as pd
import evaluate
import torch

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from data.SST2 import load_sst2_data
from data.IMDB import load_imdb_data


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }


def get_data(dataset_name, model_name):
    if dataset_name.lower() == "sst2":
        dataset, tokenizer, data_collator = load_sst2_data(
            model_name=model_name,
            max_length=128
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        text_column = "sentence"
        max_length = 128

    elif dataset_name.lower() == "imdb":
        dataset, tokenizer, data_collator = load_imdb_data(
            model_name=model_name,
            max_length=256
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        text_column = "text"
        max_length = 256

    else:
        raise ValueError("dataset_name must be either 'sst2' or 'imdb'")

    return train_dataset, eval_dataset, tokenizer, data_collator, text_column, max_length


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def safe_model_tag(model_name):
    return model_name.replace("/", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "imdb"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model_name
    model_tag = safe_model_tag(model_name)

    output_dir = f"outputs/{dataset_name}_{model_tag}"
    os.makedirs(output_dir, exist_ok=True)

    train_dataset, eval_dataset, tokenizer, data_collator, text_column, max_length = get_data(
        dataset_name=dataset_name,
        model_name=model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Training
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    # Evaluation metrics
    eval_result = trainer.evaluate()

    # Inference timing + predictions
    infer_start = time.time()
    predictions_output = trainer.predict(eval_dataset)
    inference_time = time.time() - infer_start

    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    pred_labels = np.argmax(logits, axis=-1)
    probs = softmax_np(logits)
    confidences = np.max(probs, axis=1)
    prob_label_1 = probs[:, 1]

    # Save model/tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save per-example predictions for error analysis
    texts = eval_dataset[text_column]
    pred_df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "pred_label": pred_labels,
        "confidence": confidences,
        "prob_label_1": prob_label_1,
        "correct": (pred_labels == labels).astype(int),
        "text_length_chars": [len(t) for t in texts],
        "text_length_words": [len(str(t).split()) for t in texts],
        "model_name": model_name,
        "dataset": dataset_name
    })
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # Save summary results
    results = {
        "dataset": dataset_name.upper(),
        "model_name": model_name,
        "max_length": max_length,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "weight_decay": 0.01,
        "trainable_parameters": count_parameters(model),
        "training_time_seconds": training_time,
        "inference_time_seconds_on_eval_set": inference_time,
        "num_eval_examples": len(eval_dataset),
        "avg_inference_time_per_example_seconds": inference_time / len(eval_dataset),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== Experiment Results =====")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Model: {model_name}")
    print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"F1: {eval_result['eval_f1']:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Inference Time on Eval Set: {inference_time:.2f} seconds")
    print(f"Trainable Parameters: {count_parameters(model)}")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()