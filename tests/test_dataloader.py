import unittest
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from dataloader import GPTDataset


class TestGPTDataset(unittest.TestCase):
    
    def setUp(self):
        self.file_path = 'data/tinyshakespeare.txt'
        self.seq_len = 1024
        self.dataset = GPTDataset(self.file_path, self.seq_len)
    
    def test_len(self):
        self.assertEqual(len(self.dataset), 1090)

    def test_getitem(self):
        sample_idx = 10
        sample_seq = self.dataset[sample_idx]
        self.assertEqual(len(sample_seq), self.seq_len)

    def test_dataloader(self):
        batch_size = 16
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            self.assertEqual(len(batch), batch_size)
            break
    