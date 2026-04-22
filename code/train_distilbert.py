import sys
import os
import time
import json
import numpy as np
import evaluate
import datasets
import transformers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from data.SST2 import load_sst2_data

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)


def main():
    model_name = "distilbert-base-uncased"
    max_length = 128
    output_dir = "results/distilbert_sst2_model"

    # 0. Create results directory
    os.makedirs("results", exist_ok=True)

    # 1. Load tokenized dataset
    dataset, tokenizer, data_collator = load_sst2_data(
        model_name=model_name,
        max_length=max_length
    )

    # 2. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none"
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 5. Train
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # 6. Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    print(eval_results)

    # 7. Simple inference latency test on validation set
    sample_dataset = dataset["validation"].select(
        range(min(100, len(dataset["validation"])))
    )

    infer_start = time.time()
    trainer.predict(sample_dataset)
    infer_end = time.time()

    total_infer_time = infer_end - infer_start
    avg_infer_time = total_infer_time / len(sample_dataset)

    print(f"Total inference time on {len(sample_dataset)} samples: {total_infer_time:.4f} seconds")
    print(f"Average inference time per sample: {avg_infer_time:.6f} seconds")

    # 8. Parameter count
    num_params = model.num_parameters()
    print(f"Number of parameters: {num_params}")

    # 9. Save trained model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    # 10. Save summary to txt
    summary_path = "results/distilbert_sst2_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("Dataset: glue/sst2\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Evaluation results: {eval_results}\n")
        f.write(f"Total inference time on {len(sample_dataset)} samples: {total_infer_time:.4f} seconds\n")
        f.write(f"Average inference time per sample: {avg_infer_time:.6f} seconds\n")
        f.write(f"Number of parameters: {num_params}\n")

    print(f"Summary saved to {summary_path}")

    # 11. Save config JSON
    config = {
        "model": model_name,
        "task": "SST-2 sentiment classification",
        "dataset": "glue/sst2",
        "training": {
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "weight_decay": 0.01,
            "max_length": max_length
        },
        "evaluation": {
            "metric": "accuracy",
            "final_eval_accuracy": eval_results["eval_accuracy"],
            "final_eval_loss": eval_results["eval_loss"]
        },
        "performance": {
            "training_time_seconds": training_time,
            "inference_time_100_samples_seconds": total_infer_time,
            "avg_inference_time_per_sample_seconds": avg_infer_time
        },
        "model_stats": {
            "num_parameters": num_params
        },
        "environment": {
            "transformers_version": transformers.__version__,
            "datasets_version": datasets.__version__,
            "python_version": sys.version.split()[0]
        },
        "saved_model_path": output_dir
    }

    config_path = "results/distilbert_sst2_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
