import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import argparse
import numpy as np
import evaluate

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from data.SST2 import load_sst2_data
from data.IMDB import load_imdb_data


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_data(dataset_name, model_name):
    if dataset_name.lower() == "sst2":
        dataset, tokenizer, data_collator = load_sst2_data(
            model_name=model_name,
            max_length=128
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        max_length = 128

    elif dataset_name.lower() == "imdb":
        dataset, tokenizer, data_collator = load_imdb_data(
            model_name=model_name,
            max_length=256
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        max_length = 256

    else:
        raise ValueError("dataset_name must be either 'sst2' or 'imdb'")

    return train_dataset, eval_dataset, tokenizer, data_collator, max_length


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    output_dir = f"outputs/{dataset_name}_bert_base"
    os.makedirs(output_dir, exist_ok=True)

    train_dataset, eval_dataset, tokenizer, data_collator, max_length = get_data(
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

    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    eval_result = trainer.evaluate()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

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
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== BERT-base Baseline Results =====")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Validation/Test Accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Trainable Parameters: {count_parameters(model)}")


if __name__ == "__main__":
    main()