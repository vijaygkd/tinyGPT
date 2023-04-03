import unittest
import torch
from torch.utils.data import DataLoader
from dataset import GPTDataset


# class TestGPTDataset(unittest.TestCase):

#     def test_init(self):
#         dataset = GPTDataset('tests/data.txt', seq_len=10)
#         self.assertEqual(dataset.seq_len, 10)
#         self.assertIsNotNone(dataset.tokenizer)
#         self.assertEqual(len(dataset.examples), 7)

#     def test_len(self):
#         dataset = GPTDataset('tests/data.txt', seq_len=10)
#         self.assertEqual(len(dataset), 7)

#     def test_getitem(self):
#         dataset = GPTDataset('tests/data.txt', seq_len=10)
#         input, label = dataset[0]
#         self.assertIsInstance(input, torch.Tensor)
#         self.assertIsInstance(label, torch.Tensor)
#         self.assertEqual(input.shape, torch.Size([9]))
#         self.assertEqual(label.shape, torch.Size([9]))

#     def test_dataloader(self):
#         dataset = GPTDataset('tests/data.txt', seq_len=10)
#         dataloader = DataLoader(dataset, batch_size=2)
#         for X, Y in dataloader:
#             self.assertIsInstance(X, torch.Tensor)
#             self.assertIsInstance(Y, torch.Tensor)
#             self.assertEqual(X.shape, torch.Size([2, 9]))
#             self.assertEqual(Y.shape, torch.Size([2, 9]))
