"""
Transformer components
"""
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert(d_model % n_heads==0)
        self.d_k = d_model / n_heads
        # Q, K, V
        # TODO - d_k could be user defined dimension. Allow for extension
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # query, key, value: (batch, n_heads, seq_len, d_k)
        Q = self.proj_q(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        K = self.proj_k(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        V = self.proj_v(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        # Scaled Dot-product attention: (batch, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(2,3)) / math.sqrt(self.d_k)
        # masking - mask positions to fill have value 1. Fill it with large negative number
        if mask is not None:
            attn_scores = torch.masked_fill(attn_scores, mask, value=float('-inf'))
        # softmax
        attn_scores = torch.softmax(attn_scores, dim=-1)
        # attention output: (batch, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_scores, V)
        # layer output
        attn_output = attn_output.transpose(1, 2)   # (batch, seq_len, n_heads, d_k)
        attn_output = attn_output.flatten(-2)       # (batch, seq_len, d_model)
        output = self.proj_o(attn_output)
        return output, attn_scores


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.proj_ff = nn.Linear(d_model, d_ff)
        self.proj_out = nn.Linear(d_ff, d_model)
        self.activation = nn.functional.relu

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x_ff = self.proj_ff(x)          # (batch, seq_len, d_ff)
        x_ff = self.activation(x_ff)
        output = self.proj_out(x_ff)    # (batch, seq_len, d_model)
        return output


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, p_drop):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, input, output):
        # x: (batch, seq_len, d_model)
        output = self.dropout(output)       # dropout to output of each sub-layer
        residual_output = input + output
        layernorm_output = self.layer_norm(residual_output)
        return layernorm_output


class Encoder(nn.Module):
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


# class Decoder(nn.Module):
#     def __init__(self, d_model, d_ff, n_heads, p_drop):
#         super().__init__()
#         self.self_mha = MultiHeadAttention(d_model, n_heads)
#         self.ln_self_attn = ResidualLayerNorm(d_model, p_drop)
#         self.x_mha = MultiHeadAttention(d_model, n_heads)
#         self.ln_x_attn = ResidualLayerNorm(d_model, p_drop)
#         self.ff = PositionWiseFeedForward(d_model, d_ff)
#         self.ln_ff = ResidualLayerNorm(d_model, p_drop)

#     def forward(self, x, encoder_output):
#         # x: (batch, seq_len, d_model)
#         # encoder_output: (batch, seq_len, d_model)



 
