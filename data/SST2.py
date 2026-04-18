from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, max_length=128)

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.remove_columns(["sentence", "idx"])
dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer) # this is the tokenized data