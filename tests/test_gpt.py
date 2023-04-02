import torch
import unittest
from torch import nn
from GPT import GPT, Decoder


class TestGPT(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 3
        self.vocab_size = 10
        self.n_blocks = 2
        self.d_model = 4
        self.d_ff = 16
        self.n_heads = 2
        self.p_drop = 0.1

        self.gpt = GPT(
            n_blocks=self.n_blocks,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            p_drop=self.p_drop,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len
        )

    def test_shape_of_output(self):
        x = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        mask = torch.zeros((self.batch_size, self.seq_len, self.seq_len), dtype=torch.float)
        logits, decoder_attns = self.gpt(x, mask)

        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(len(decoder_attns), self.n_blocks)

    def test_decoder_output_shape(self):
        decoder = Decoder(self.d_model, self.d_ff, self.n_heads, self.p_drop)
        x = torch.zeros((self.batch_size, self.seq_len, self.d_model), dtype=torch.float)
        output, attn_scores = decoder(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(attn_scores.shape, (self.batch_size, self.n_heads, self.seq_len, self.seq_len))

    def test_forward(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        mask = torch.zeros((self.batch_size, self.seq_len, self.seq_len))
        logits, decoder_attns = self.gpt(x, mask)

        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(len(decoder_attns), self.n_blocks)
