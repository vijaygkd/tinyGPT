"""
GPT model definition
"""
import math
import torch
from torch import nn
from tinytransformer import MultiHeadAttention, PositionWiseFeedForward, ResidualLayerNorm, SharedEmbeddingLayer


class GPT(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, p_drop, vocab_size, seq_len, device='cpu'):
        super().__init__()
        
        self.d_model = d_model
        # decoder stack
        decoder_blocks =  []
        for i in range(n_blocks):
            decoder = GPTDecoder(d_model, d_ff, n_heads, p_drop, device)
            decoder_blocks.append(decoder)
        self.decoder_stack = nn.ModuleList(decoder_blocks)
        # shared weights: token embeddings & decoder output
        self.token_embedding = SharedEmbeddingLayer(vocab_size, d_model)
        # TODO - positional embedding logic
        self.positional_embedding = nn.Embedding(seq_len, d_model)   # learned embedding
        self.position_ids = torch.arange(seq_len, requires_grad=False).to(device)   # position ids: [0,1,2...,n]
        self.emb_dropout = nn.Dropout(p_drop)
        # future mask
        self.future_mask = torch.tril(torch.ones((seq_len, seq_len), requires_grad=False).to(device))
        # model to gpu
        self.to(device)
        

    # TODO - implement padding mask and combine it with training mask
    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        decoder_attns = []
        token_emb = self.token_embedding(x)                         # (batch, seq_len, d_model)
        pos_embd = self.positional_embedding(self.position_ids)     # (batch, seq_len, d_model)
        input_emb = token_emb + pos_embd                            # (batch, seq_len, d_model)
        input_emb = self.emb_dropout(input_emb)
        # decoder stack
        dec_out = input_emb
        for decoder in self.decoder_stack:
            dec_out, attn = decoder(dec_out, self.future_mask)      # dec_out: (batch, seq_len, d_model)
            decoder_attns.append(attn)                              # attn: (batch, n_heads, seq_len, seq_len)
        # output
        logits = self.token_embedding(dec_out, mode='linear')       # (batch, seq_len, vocab_size)
        return logits, decoder_attns


class GPTDecoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, p_drop, device='cpu'):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads).to(device)
        self.ln_attn = ResidualLayerNorm(d_model, p_drop).to(device)
        self.ff = PositionWiseFeedForward(d_model, d_ff).to(device)
        self.ln_ff = ResidualLayerNorm(d_model, p_drop).to(device)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        attn_out, attn_scores = self.mha(x, mask)
        attn_out = self.ln_attn(x, attn_out)
        ff_out = self.ff(attn_out)
        ff_out = self.ln_ff(attn_out, ff_out)
        return ff_out, attn_scores
