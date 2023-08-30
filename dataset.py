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
    """
    def __init__(self, tokenizer, file_path, seq_len):
        self.seq_len = seq_len   
        split_size = seq_len + 1       # right shit input by 1 position to create labels for training
        self.tokenizer = tokenizer 
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize the text and split into sequences of length `seq_len`
        self.tokens = self.tokenizer.encode(self.text)
        self.examples = [self.tokens[i:i+split_size] for i in range(0, len(self.tokens), split_size)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Convert the example to input and label tensors. 
        example = self.examples[idx]
        example = torch.tensor(example, dtype=torch.long)
        input = example[:-1]
        # Label right shifted by 1.
        label = example[1:]
        return (input, label)


def pad_seq_fn(batch, pad_id):
    # Pad the examples to the same length # GPT2 tokenizer <eos> as 50256
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)    
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_id)
    return inputs, labels


#GPT2TokenizerFast.from_pretrained('gpt2')
# Returns text encoding using byte-pair encoding as input and label pairs.
# Label is the right shifted input by 1 position to next-token prediction task.
# Eg.Text: "I love tennis!" Input: ['I', 'love', 'tennis'] and label: ['love', 'tennis', '!']


########################
# Char level tokenizer and dataset
########################
class CharTokenizer():
    """
    Tokenizer is character level ASCII (0-127). Use eos_token at start and in between samples.
    Label is right shifted input by 1 position to next-token prediction task.
    Eg.Text: "I love tennis!" Input: [<eos>, 'I', ' ', 'l', 'o', 'v', 'e', ' ', 't', 'e', 'n', 'n', 'i', 's', '!', <eos>]
    """
    def __init__(self):
        self.eos_token_id = 127
        self.eos_token = chr(self.eos_token_id)
        self.vocab_size = 128

    def tokenize(self, text):
        tokens = [self.eos_token] + list(text) + [self.eos_token]
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [ord(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [chr(id) for id in ids]
    
    def encode(self, text):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids
    
    def decode(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        text = ''.join(tokens)
        return text
    


if __name__ == '__main__':
    tkz = GPT2TokenizerFast.from_pretrained('gpt2')
    chr_tkz = CharTokenizer()

    dataset = GPTDataset(tkz, 'tests/data.txt', seq_len=10)
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=lambda x: pad_seq_fn(x, chr_tkz.eos_token_id))

    for X, y in dataloader:
        print(X)
        print(y)
    