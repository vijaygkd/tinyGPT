import torch
from transformer import MultiHeadAttention

def test_MultiHeadAttention():
    d_model = 64
    n_heads = 8
    seq_len = 10
    batch_size = 4
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.zeros(batch_size, n_heads, seq_len, seq_len).bool()

    mha = MultiHeadAttention(d_model, n_heads)
    output, attn_scores = mha(x, mask)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape: {(batch_size, seq_len, d_model)}, got: {output.shape}"
    assert attn_scores.shape == (batch_size, n_heads, seq_len, seq_len), f"Expected attn_scores shape: {(batch_size, n_heads, seq_len, seq_len)}, got: {attn_scores.shape}"

    # Test with mask
    mask[0, 0, 0, 1] = 1
    output, attn_scores = mha(x, mask)
    assert attn_scores[0, 0, 0, 1] == 0, f"Expected masked value to be 0, got: {attn_scores[0, 0, 0, 1]}"

if __name__ == '__main__':
    test_MultiHeadAttention()