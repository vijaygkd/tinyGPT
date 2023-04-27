"""
HF dataset
Loading python data science code code completion dataset from HuggingFace
https://huggingface.co/learn/nlp-course/chapter7/6
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

from datasets import load_dataset, DatasetDict


def get_codeparrot_dataset(seq_len, split="valid"):
    """
    Load the codeparrot dataset from HuggingFace and returns a tokenized dataset
    seq_len: int, max sequence length
    split: str, one of "train", "valid"
    """
    # Tokenize the dataset
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    if split == "train":
        ds = load_dataset("huggingface-course/codeparrot-ds-valid", split="train")
    elif split == "valid":
        ds = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
    # Tokenize the dataset
    tokens = ds.map(lambda e: tokenizer(e["content"], 
                                         padding="max_length", 
                                         truncation=True, 
                                         max_length=seq_len + 1,            # to offset labels from input by 1 position
                                         return_overflowing_tokens=True,
                    ), batched=True, remove_columns=ds.column_names)
    return tokens


def prepare_input_labels(batch):
    inputs, labels = [], []
    for item in batch:
        ids = item['input_ids']
        inputs.append(ids[:-1])
        labels.append(ids[1:])
    return torch.tensor(inputs, dtype=torch.long) , torch.tensor(labels, dtype=torch.long)


if __name__ == "__main__":
    tokens = get_codeparrot_dataset(seq_len=50, split="valid")
    dataloader = DataLoader(tokens, batch_size=4, collate_fn=prepare_input_labels)
    for X, y in dataloader:
        print(X)
        print(y)
