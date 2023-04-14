import unittest
import torch
from torch.utils.data import DataLoader
from dataset import GPTDataset, pad_seq_fn


class TestGPTDataset(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        self.dataset = GPTDataset('tests/data.txt', seq_len=10)

    def test_getitem(self):
        input, label = self.dataset[0]
        self.assertIsInstance(input, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(input.shape, torch.Size([10]))
        self.assertEqual(label.shape, torch.Size([10]))

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=2, collate_fn=pad_seq_fn)
        for X, Y in dataloader:
            self.assertIsInstance(X, torch.Tensor)
            self.assertIsInstance(Y, torch.Tensor)
            self.assertEqual(X.shape, torch.Size([2, 10]))
            self.assertEqual(Y.shape, torch.Size([2, 10]))


def test_pad_seq_fn():
    batch = [(torch.tensor([1, 2]), torch.tensor([2, 3])), 
             (torch.tensor([4, 5, 6]), torch.tensor([5, 6, 7]))]
    inputs, labels = pad_seq_fn(batch)
    assert inputs.shape == (2, 3)
    assert labels.shape == (2, 3)
    # assert first batch is padded with 0
    assert torch.equal(inputs[0], torch.tensor([1, 2, 0]))
    assert torch.equal(labels[0], torch.tensor([2, 3, 0]))

