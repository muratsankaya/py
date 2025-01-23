import torch
from src.transformer_v2 import FeedForward


def test_feed_forward():
    ff = FeedForward()
    x = torch.randn(2, 2, 512)
    assert ff(x).size() == (2, 2, 512), "The input input dimensions should not change"
