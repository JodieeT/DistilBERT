from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256
    )

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer) # this is the tokenized data