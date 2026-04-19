import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import numpy as np
import evaluate

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from data.SST2 import load_sst2_data
# 如果以后要跑 IMDb，就改成：
# from IMDB import load_imdb_data


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)


def main():
    # 1. Load tokenized dataset
    dataset, tokenizer, data_collator = load_sst2_data(
        model_name="distilbert-base-uncased",
        max_length=128
    )

    # 2. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir="./distilbert-sst2",
        evaluation_strategy="epoch",
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
        tokenizer=tokenizer,
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
    sample_dataset = dataset["validation"].select(range(min(100, len(dataset["validation"]))))

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

    # 9. Save summary to txt
    with open("distilbert_sst2_results.txt", "w") as f:
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Evaluation results: {eval_results}\n")
        f.write(f"Total inference time on {len(sample_dataset)} samples: {total_infer_time:.4f} seconds\n")
        f.write(f"Average inference time per sample: {avg_infer_time:.6f} seconds\n")
        f.write(f"Number of parameters: {num_params}\n")


if __name__ == "__main__":
    main()
