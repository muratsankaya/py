import torch
import pytest
from src.transformer_v1 import Decoder, get_pad_mask, get_attention_mask


@pytest.mark.parametrize(
    "batch_size, number_of_heads, seq_len, d_model",
    [(2, 1, 2, 8), (16, 16, 64, 512), (100, 4, 100, 1024)],
)
def test_decoder(batch_size, number_of_heads, seq_len, d_model):

    decoder = Decoder(
        attention_mask=get_attention_mask(seq_len, torch.float32),
        number_of_heads=number_of_heads,
        d_model=d_model,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_x = torch.randn(batch_size, seq_len, d_model)
    assert decoder(
        x,
        encoder_x,
        decoder_pad_mask=get_pad_mask(
            x=torch.randint(0, 100, (batch_size, seq_len)),
            h=number_of_heads,
            batch_size=batch_size,
            seq_len=seq_len,
        ),
        encoder_pad_mask=get_pad_mask(
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
