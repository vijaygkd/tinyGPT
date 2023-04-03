"""
Prepare data from training!

TODO:
1. function to process input text
2. Byte pair encoding
3. Masking - Padding mask, subsequent mask
4. Dataloader class
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast


class GPTDataset(Dataset):
    """
    Read a text file and break in into chunks of size seq_len.
    Returns text encoding using byte-pair encoding.
    """
    def __init__(self, file_path, seq_len=512):
        self.seq_len = seq_len
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize the text and split into sequences of length `seq_len`
        self.tokens = self.tokenizer.encode(self.text)
        self.examples = [self.tokens[i:i+self.seq_len] for i in range(0, len(self.tokens), self.seq_len)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Convert the example to a tensor
        example = torch.tensor(self.examples[idx], dtype=torch.long)
        input = example[:-1]
        label = example[1:]
        return (input, label)


if __name__ == '__main__':
    dataset = GPTDataset('tests/data.txt', seq_len=10)
    dataloader = DataLoader(dataset, batch_size=2)
    for X, y in dataloader:
        print(X)
        print(y)