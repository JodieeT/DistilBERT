import sys
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def has_negation(text):
    neg_words = ["not", "no", "never", "n't"]
    text = text.lower()
    return int(any(word in text for word in neg_words))


def main():
    model_dir = "results/distilbert_sst2_model"
    output_csv = "results/distilbert_predictions.csv"
    max_length = 128

    os.makedirs("results", exist_ok=True)

    # 1. Load raw SST-2 dataset
    raw_dataset = load_dataset("glue", "sst2")
    val_texts = raw_dataset["validation"]["sentence"]

    # 2. Load tokenizer from saved fine-tuned model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            max_length=max_length
        )

    tokenized_dataset = raw_dataset.map(preprocess, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset.set_format("torch")

    # 3. Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # 4. Create trainer for prediction
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    # 5. Predict on validation set
    predictions = trainer.predict(tokenized_dataset["validation"])

    logits = predictions.predictions
    labels = predictions.label_ids

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    # 6. Build dataframe
    df = pd.DataFrame({
        "id": range(len(val_texts)),
        "text": val_texts,
        "true_label": labels,
        "distilbert_pred": preds,
        "distilbert_confidence": confidence,
    })

    df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
    df["correct"] = (df["true_label"] == df["distilbert_pred"]).astype(int)
    df["has_negation"] = df["text"].apply(has_negation)

    # 7. Clean dtypes
    df["true_label"] = df["true_label"].astype(int)
    df["distilbert_pred"] = df["distilbert_pred"].astype(int)
    df["distilbert_confidence"] = df["distilbert_confidence"].astype(float)
    df["text_length"] = df["text_length"].astype(int)
    df["correct"] = df["correct"].astype(int)
    df["has_negation"] = df["has_negation"].astype(int)

    # 8. Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # 9. Print quick summary
    accuracy = (df["true_label"] == df["distilbert_pred"]).mean()
    print(f"Validation accuracy from saved predictions: {accuracy:.4f}")
    print(df.head())


if __name__ == "__main__":
    main()
