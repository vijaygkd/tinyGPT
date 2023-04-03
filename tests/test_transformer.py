import torch
from tinytransformer import MultiHeadAttention, PositionWiseFeedForward, ResidualLayerNorm, Encoder

def test_MultiHeadAttention():
    d_model = 64
    n_heads = 8
    seq_len = 10
    batch_size = 4
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, n_heads, seq_len, seq_len).bool()

    mha = MultiHeadAttention(d_model, n_heads)
    output, attn_scores = mha(x, mask)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape: {(batch_size, seq_len, d_model)}, got: {output.shape}"
    assert attn_scores.shape == (batch_size, n_heads, seq_len, seq_len), f"Expected attn_scores shape: {(batch_size, n_heads, seq_len, seq_len)}, got: {attn_scores.shape}"

    # Test with mask
    mask[0, 0, 0, 1] = 0
    output, attn_scores = mha(x, mask)
    assert attn_scores[0, 0, 0, 1] == 0, f"Expected masked value to be 0, got: {attn_scores[0, 0, 0, 1]}"

    # Test subsequent mask for teacher training
    mask = torch.tril(torch.ones(seq_len, seq_len))
    output, attn_scores = mha(x, mask)
    # check if all elements are 0 after zero-ing out lower and diagonal elements
    bottom_zero_out = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    assert (attn_scores * bottom_zero_out).sum() == 0


def test_PositionWiseFeedForward():
    d_model = 64
    d_ff = 128
    seq_len = 10
    batch_size = 4
    x = torch.randn(batch_size, seq_len, d_model)

    pwff = PositionWiseFeedForward(d_model, d_ff)
    output = pwff(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape: {(batch_size, seq_len, d_model)}, got: {output.shape}"

def test_ResidualLayerNorm():
    p_drop = 0.1
    seq_len = 10
    batch_size = 4
    d_model = 64
    input = torch.randn(batch_size, seq_len, d_model)
    output = torch.randn(batch_size, seq_len, d_model)

    rln = ResidualLayerNorm(d_model, p_drop)
    layernorm_output = rln(input, output)

    assert layernorm_output.shape == (batch_size, seq_len, d_model), f"Expected output shape: {(batch_size, seq_len, d_model)}, got: {layernorm_output.shape}"

def test_Encoder():
    d_model = 64
    d_ff = 128
    n_heads = 8
    p_drop = 0.1
    seq_len = 10
    batch_size = 4
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, n_heads, seq_len, seq_len)
    mask[0, 0, 0, 1] = 0

    encoder = Encoder(d_model, d_ff, n_heads, p_drop)
    ff_out, attn_scores = encoder(x, mask)

    assert ff_out.shape == (batch_size, seq_len, d_model), f"Expected output shape: {(batch_size, seq_len, d_model)}, got: {ff_out.shape}"
    assert attn_scores[0, 0, 0, 1] == 0, f"Expected masked value to be 0, got: {attn_scores[0, 0, 0, 1]}"
