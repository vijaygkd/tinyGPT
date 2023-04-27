"""
Read a text file and break in into chunks of size seq_len.

TODO:
3. Masking - Padding mask, subsequent mask. return these along with input and output
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast


class GPTDataset(Dataset):
    """
    Read a text file and break in into chunks of size seq_len.
    Returns text encoding using byte-pair encoding as input and label pairs.
    Label is the right shifted input by 1 position to next-token prediction task.
    Eg.Text: "I love tennis!" Input: ['I', 'love', 'tennis'] and label: ['love', 'tennis', '!']
    """
    def __init__(self, file_path, seq_len):
        self.seq_len = seq_len   
        split_size = seq_len + 1       # right shit input by 1 position to create labels for training
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize the text and split into sequences of length `seq_len`
        self.tokens = self.tokenizer.encode(self.text)
        self.examples = [self.tokens[i:i+split_size] for i in range(0, len(self.tokens), split_size)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Convert the example to input and label tensors. 
        example = torch.tensor(self.examples[idx], dtype=torch.long)
        input = example[:-1]
        # Label right shifted by 1.
        label = example[1:]
        return (input, label)


def pad_seq_fn(batch):
    # Pad the examples to the same length
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=50256)    # GPT2 tokenizer <eos> as 50256
    labels = pad_sequence(labels, batch_first=True, padding_value=50256)
    return inputs, labels


if __name__ == '__main__':
    dataset = GPTDataset('tests/data.txt', seq_len=10)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=pad_seq_fn)
    for X, y in dataloader:
        print(X)
        print(y)