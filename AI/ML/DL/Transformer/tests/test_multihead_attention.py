import torch
import pytest
from src.transformer import MultiHead, get_pad_mask


# Test the MultiHead class
def test_multihead_attention():
    # Input configurations
    batch_size = 2
    seq_len = 4
    d_model = 16
    h = 4

    # Generate random tensors for Q, K, V
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    # Initialize MultiHead module
    multi_head = MultiHead(h, d_model)

    # Compute multi-head attention
    output = multi_head(
        Q,
        K,
        V,
        pad_mask=get_pad_mask(
            x=torch.randint(0, 100, (batch_size, seq_len)),
            h=h,
            batch_size=batch_size,
            seq_len=seq_len,
        ),
    )

    # Check output shape
    assert output.size() == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.size()}"

    # Check output type
    assert isinstance(output, torch.Tensor), "Output is not a PyTorch tensor"

    # Check if output is differentiable
    assert output.requires_grad, "Output does not require gradients"


@pytest.mark.parametrize("h, d_model", [(1, 8), (2, 16), (4, 32)])
def test_multihead_attention_with_various_heads(h, d_model):
    # Input configurations
    batch_size = 2
    seq_len = 4

    # Generate random tensors for Q, K, V
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    # Initialize MultiHead module
    multi_head = MultiHead(h, d_model)

    # Compute multi-head attention
    output = multi_head(
        Q,
        K,
        V,
        pad_mask=get_pad_mask(
            x=torch.randint(0, 100, (batch_size, seq_len)),
            h=h,
            batch_size=batch_size,
            seq_len=seq_len,
        ),
    )

    # Check output shape
    assert output.size() == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.size()}"
