import torch
import pytest
from src.transformer import Encoder, get_pad_mask


@pytest.mark.parametrize(
    "batch_size, number_of_heads, seq_len, d_model",
    [(2, 1, 2, 8), (16, 16, 64, 512), (100, 4, 100, 1024)],
)
def test_encoder(batch_size, number_of_heads, seq_len, d_model):

    encoder = Encoder(
        number_of_heads=number_of_heads,
        d_model=d_model,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    assert encoder(
        x,
        pad_mask=get_pad_mask(
            x=torch.randint(0, 100, (batch_size, seq_len)),
            h=number_of_heads,
            batch_size=batch_size,
            seq_len=seq_len,
        ),
    ).size() == (
        batch_size,
        seq_len,
        d_model,
    ), "input and output sizes must equal"
