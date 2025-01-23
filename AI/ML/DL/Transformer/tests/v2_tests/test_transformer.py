import torch
import pytest
from src.transformer_v2 import Transformer
from tests.helper_functions import print_green


def get_pad_mask_transformer(x):
    return torch.where(x == 0, 0, 1)


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, n, number_of_heads, vocab_size",
    [
        (2, 2, 8, 1, 2, 10),  # Small vocab size for simple test
        (16, 64, 512, 2, 8, 10000),  # Larger vocab size for more complex scenarios
    ],
)
def test_transformer(batch_size, seq_len, d_model, n, number_of_heads, vocab_size):
    """
    Test the Transformer model to ensure it outputs the correct shape based on vocab_size.
    """
    # Ensure that d_model is divisible by number_of_heads
    assert (
        d_model % number_of_heads == 0
    ), "d_model must be divisible by number_of_heads"

    # Instantiate the Transformer model with vocab_size
    transformer = Transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n=n,
        number_of_heads=number_of_heads,
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )

    # Create random input tensors for encoder and decoder
    encoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    decoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))

    # Create pad masks
    encoder_pad_mask = get_pad_mask_transformer(encoder_x)
    decoder_pad_mask = get_pad_mask_transformer(decoder_x)

    # Perform a forward pass
    output = transformer(encoder_x, decoder_x, encoder_pad_mask, decoder_pad_mask)

    # Assert that the output shape matches the expected shape with vocab_size
    assert output.size() == (batch_size, seq_len, vocab_size), (
        f"Expected output shape {(batch_size, seq_len, vocab_size)}, "
        f"but got {output.size()}"
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, n, number_of_heads, vocab_size",
    [
        (2, 2, 8, 1, 2, 10),
        (16, 64, 512, 2, 8, 10000),
    ],
)
def test_transformer_gradients_flow(
    batch_size, seq_len, d_model, n, number_of_heads, vocab_size
):
    """
    Test that gradients flow through all parameters of the Transformer.
    """
    transformer = Transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n=n,
        number_of_heads=number_of_heads,
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )

    # Create random integer inputs
    encoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    decoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))

    # Create pad masks
    encoder_pad_mask = get_pad_mask_transformer(encoder_x)
    decoder_pad_mask = get_pad_mask_transformer(decoder_x)

    # Forward pass
    output = transformer(encoder_x, decoder_x, encoder_pad_mask, decoder_pad_mask)
    loss = output.sum()

    # Backpropagate
    loss.backward()

    # Check gradients in parameters
    no_grad_params = []
    for name, param in transformer.named_parameters():
        if param.grad is None:
            no_grad_params.append(name)

    assert (
        len(no_grad_params) == 0
    ), f"Some parameters did not receive gradients: {no_grad_params}"

    print_green("All parameters received gradients successfully.")


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, n, number_of_heads, vocab_size",
    [
        (2, 2, 8, 1, 2, 10),
        (16, 64, 512, 2, 8, 10000),
    ],
)
def test_transformer_parameter_count(
    batch_size, seq_len, d_model, n, number_of_heads, vocab_size
):
    """
    Test that the Transformer model has parameters.
    """
    # Instantiate the Transformer model
    transformer = Transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n=n,
        number_of_heads=number_of_heads,
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )

    # Check that the model has parameters
    num_params = sum(p.numel() for p in transformer.parameters())
    assert num_params > 0, "Transformer model has no parameters"


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, n, number_of_heads, vocab_size",
    [
        (2, 2, 8, 1, 2, 10),
        (16, 64, 512, 2, 8, 10000),
    ],
)
def test_transformer_forward_output_values(
    batch_size, seq_len, d_model, n, number_of_heads, vocab_size
):
    """
    Test that the Transformer's forward pass (with softmax) produces valid probabilities.
    """
    transformer = Transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n=n,
        number_of_heads=number_of_heads,
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )

    encoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    decoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))

    # Create pad masks
    encoder_pad_mask = get_pad_mask_transformer(encoder_x)
    decoder_pad_mask = get_pad_mask_transformer(decoder_x)

    # Perform a forward pass with apply_softmax=True
    output = transformer(
        encoder_x, decoder_x, encoder_pad_mask, decoder_pad_mask, apply_softmax=True
    )

    # Check if output values are between 0 and 1
    assert torch.all(output >= 0) and torch.all(
        output <= 1
    ), "Output values are not in [0, 1]"

    # Check that each softmax vector sums to 1
    sums = output.sum(dim=2)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-6
    ), "Softmax outputs do not sum to 1"

    assert output.size() == (batch_size, seq_len, vocab_size), (
        f"Expected output shape {(batch_size, seq_len, vocab_size)}, "
        f"but got {output.size()}"
    )


@pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.5])
def test_transformer_dropout_parameter(dropout_p):
    """
    Test the Transformer model's behavior under various dropout probabilities.
    """
    batch_size = 4
    seq_len = 8
    d_model = 64
    n = 2
    number_of_heads = 8
    vocab_size = 100

    transformer = Transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n=n,
        number_of_heads=number_of_heads,
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        dropout_p=dropout_p,
    )

    # Create deterministic input
    torch.manual_seed(0)
    encoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    decoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))

    # Create pad masks
    encoder_pad_mask = get_pad_mask_transformer(encoder_x)
    decoder_pad_mask = get_pad_mask_transformer(decoder_x)

    # Forward pass multiple times
    outputs = []
    for _ in range(3):
        out = transformer(encoder_x, decoder_x, encoder_pad_mask, decoder_pad_mask)
        outputs.append(out.detach().clone())

    if dropout_p == 0.0:
        # Outputs should be identical if dropout is 0.0
        assert torch.allclose(outputs[0], outputs[1]) and torch.allclose(
            outputs[1], outputs[2]
        ), "Outputs differ across runs even though dropout is 0.0."
    else:
        # With dropout > 0, it's unlikely all outputs are identical
        all_identical = True
        for i in range(1, len(outputs)):
            if not torch.allclose(outputs[0], outputs[i]):
                all_identical = False
                break

        assert (
            not all_identical
        ), "All outputs are identical across multiple runs even though dropout is applied."
