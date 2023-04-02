"""
GPT model definition
"""

import torch
from torch import nn
from transformer import MultiHeadAttention, PositionWiseFeedForward, ResidualLayerNorm


class GPT(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, p_drop, vocab_size, seq_len):
        super().__init__()
        self.decoder_stack =  []
        for i in range(n_blocks):
            decoder = GPTDecoder(d_model, d_ff, n_heads, p_drop)
            self.decoder_stack.append(decoder)
        self.proj_output = nn.Linear(d_model, vocab_size)
        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # TODO - positional embedding logic
        self.positional_embedding = nn.Embedding(seq_len, d_model)   # learned embedding
        self.position_ids = torch.arange(seq_len)   # position ids: [0,1,2...,n]
        self.emb_dropout = nn.Dropout(p_drop)


    def forward(self, x, mask):
        # x: (batch, seq_len)
        decoder_attns = []
        token_emb = self.token_embedding(x)                         # (batch, seq_len, d_model)
        pos_embd = self.positional_embedding(self.position_ids)     # (batch, seq_len, d_model)
        input_emb = token_emb + pos_embd                            # (batch, seq_len, d_model)
        input_emb = self.emb_dropout(input_emb)
        # decoder stack
        dec_out = input_emb
        for decoder in self.decoder_stack:
            dec_out, attn = decoder(dec_out, mask)  # (batch, seq_len, d_model)
            decoder_attns.append(attn)              # (batch, n_heads, seq_len, seq_len)
        # output
        logits = self.proj_output(dec_out)          # (batch, seq_len, vocab_size)
        return logits, decoder_attns


class GPTDecoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, p_drop):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln_attn = ResidualLayerNorm(d_model, p_drop)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.ln_ff = ResidualLayerNorm(d_model, p_drop)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        attn_out, attn_scores = self.mha(x, mask)
        attn_out = self.ln_attn(x, attn_out)
        ff_out = self.ff(attn_out)
        ff_out = self.ln_ff(attn_out, ff_out)
        return ff_out, attn_scores
