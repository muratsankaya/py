"""
This file imitiates the pad masks created with tokenizer_helinski_nlp_opus
"""

import torch


def get_pad_mask(x: torch.Tensor, h: int, batch_size: int, seq_len: int):
    assert x.size() == (
        batch_size,
        seq_len,
    ), f"expected x to have size: {(batch_size, seq_len)}, got: {x.size()} instead."
    return (
        torch.where(x == 0, 0, 1)
        .view(batch_size, 1, 1, seq_len)
        .expand(-1, h, seq_len, -1)
        .reshape(batch_size * h, seq_len, seq_len)
    )
