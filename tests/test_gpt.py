import unittest
import torch
from torch import nn
from GPT import GPT, GPTDecoder


class TestGPT(unittest.TestCase):
    def test_gpt(self):
        # model
        model = GPT(n_blocks=2, d_model=512, d_ff=2048, n_heads=8, p_drop=0.1, vocab_size=1000, seq_len=128)
        # input
        x = torch.randint(0, 1000, (2, 128))
        # forward
        logits, decoder_attns = model(x)
        # test
        self.assertEqual(logits.shape, (2, 128, 1000))
        self.assertEqual(len(decoder_attns), 2)
        self.assertEqual(decoder_attns[0].shape, (2, 8, 128, 128))
        self.assertEqual(decoder_attns[1].shape, (2, 8, 128, 128))
        # check future mask works
        final_attn = decoder_attns[-1]
        bottom_zero_out = torch.triu(torch.ones((128, 128)), diagonal=1)    # zero out lower and diagonal elements
        assert (final_attn * bottom_zero_out).sum() == 0
    
    def test_gpt_with_padding_mask_TODO(self):
        # model
        model = GPT(n_blocks=2, d_model=512, d_ff=2048, n_heads=8, p_drop=0.1, vocab_size=1000, seq_len=128)
        # input
        x = torch.randint(0, 1000, (2, 128))
        # forward
        logits, decoder_attns = model(x)
        # test output with padding mask
        # TODO 
        self.assertEqual(logits.shape, (2, 128, 1000))
        self.assertEqual(len(decoder_attns), 2)
        self.assertEqual(decoder_attns[0].shape, (2, 8, 128, 128))
        self.assertEqual(decoder_attns[1].shape, (2, 8, 128, 128))
        # check future mask works
        final_attn = decoder_attns[-1]
        bottom_zero_out = torch.triu(torch.ones((128, 128)), diagonal=1)    # zero out lower and diagonal elements
        assert (final_attn * bottom_zero_out).sum() == 0


class TestGPTDecoder(unittest.TestCase):
    def test_gpt_decoder(self):
        # model
        model = GPTDecoder(d_model=512, d_ff=2048, n_heads=8, p_drop=0.1)
        # input
        x = torch.randn(2, 128, 512)
        mask = torch.tril(torch.ones((128, 128)))
        # forward
        out, attn_scores = model(x, mask)
        # test
        self.assertEqual(out.shape, (2, 128, 512))
        self.assertEqual(attn_scores.shape, (2, 8, 128, 128))
