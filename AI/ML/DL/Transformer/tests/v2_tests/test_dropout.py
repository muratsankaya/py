import pytest
import torch
from src.transformer_v2 import dropout


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model", [(4, 4, 4), (20, 16, 16), (200, 512, 1024)]
)
def test_dropout(batch_size, seq_len, d_model):
    x = torch.randn((batch_size, seq_len, d_model))

    # Well I don't know how to test the exact behaviour as
    # there is no exact behaviour.

    # When p is 1 all the values in the output tensor should be 0
    assert torch.all(dropout(x, p=1) == 0), "when p is 1 all the entries should be 0!"

    # Sanity check input and output tensors should have the same size
    assert (
        x.size() == dropout(x, p=torch.rand(1).item()).size()
    ), "input and output tensors should have the same size"


def test_dropout_statistical_behavior():
    """
    In theory this test can fail, but in practice
    I highly doubt it will fail unless there is a mistake in the implementation of dropout
    """
    # We want to test a scenario where p = 0.7, and verify that on average
    # at least 35% of the entries are zeroed out.
    p = 0.7
    batch_size, seq_len, d_model = 32, 32, 32
    x = torch.randn((batch_size, seq_len, d_model))

    num_trials = 20
    zeroed_ratios = []
    for _ in range(num_trials):
        out = dropout(x, p=p)
        zeroed_count = (out == 0).sum().item()
        total_count = out.numel()
        zeroed_ratio = zeroed_count / total_count
        zeroed_ratios.append(zeroed_ratio)

    # Check the average zeroed ratio
    avg_zeroed_ratio = sum(zeroed_ratios) / num_trials
    # We expect at least half of the requested dropout fraction (35% of 70%)
    # This is a relaxed threshold to account for randomness.
    assert (
        avg_zeroed_ratio > 0.35
    ), f"Expected at least 35% zeroed on average, but got {avg_zeroed_ratio*100:.2f}%"
