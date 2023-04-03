"""
Prepare data from training!

TODO:
1. function to process input text
2. Byte pair encoding
3. Masking - Padding mask, subsequent mask
4. Dataloader class
"""

from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast


class GPTDataset(Dataset):
    def __init__(self, file_path, seq_len) -> None:
        super().__init__()
        with open(file_path, "r") as file:
            text = file.read()
        self.data = [text[i: i+seq_len] for i in range(0, len(text), seq_len)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


