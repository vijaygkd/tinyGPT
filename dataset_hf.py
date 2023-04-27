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


def get_codeparrot_dataset(seq_len, split="validation"):
    """
    Load the codeparrot dataset from HuggingFace
    seq_len: int, max sequence length
    split: str, one of "train", "validation"
    """
    # Load the dataset
    ds = load_dataset("huggingface-course/codeparrot-ds-valid", split=split)

    # Tokenize the dataset
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    token_outputs = tokenizer(
        ds[:1]["content"],
        padding="max_length",
        truncation=True,
        max_length=seq_len + 1,             # to offset labels from input by 1 position
        return_overflowing_tokens=True,
        return_length=True,
        return_tensors="pt",
    )
    return token_outputs


def prepare_input_labels(batch):
    inputs, labels = [], []
    for item in batch:
        inputs.append(item.ids[:-1])
        labels.append(item.ids[1:])
    return inputs, labels


if __name__ == "__main__":
    op = get_codeparrot_dataset(seq_len=50, split="validation")
    dataloader = DataLoader(op, batch_size=4, collate_fn=prepare_input_labels)
    for X, y in dataloader:
        print(X)
        print(y)
