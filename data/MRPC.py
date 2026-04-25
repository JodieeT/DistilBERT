from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)


def load_mrpc_data(model_name="distilbert-base-uncased", max_length=128):
    """Load GLUE MRPC paraphrase-identification dataset with sentence-pair tokenization."""
    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer)
    return dataset, tokenizer, data_collator
