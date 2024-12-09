import torch
import pytest
from src.transformer import Transformer


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

    Parameters:
    - batch_size (int): Number of samples in a batch.
    - seq_len (int): Length of the input sequences.
    - d_model (int): Dimension of the model (embedding size).
    - n (int): Number of encoder and decoder layers.
    - number_of_heads (int): Number of attention heads.
    - vocab_size (int): Size of the target vocabulary.
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

    # Perform a forward pass
    output = transformer(encoder_x, decoder_x)

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

    # Create random integer inputs. No requires_grad=True here.
    encoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    decoder_x = torch.randint(0, vocab_size, size=(batch_size, seq_len))

    # Forward pass
    output = transformer(encoder_x, decoder_x)
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

    print("All parameters received gradients successfully.")


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
    Test that the Transformer model has the expected number of parameters.

    Parameters:
    - batch_size (int): Number of samples in a batch.
    - seq_len (int): Length of the input sequences.
    - d_model (int): Dimension of the model (embedding size).
    - n (int): Number of encoder and decoder layers.
    - number_of_heads (int): Number of attention heads.
    - vocab_size (int): Size of the target vocabulary.
    """
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

    # Calculate expected number of parameters
    # This is a placeholder; actual calculation depends on Encoder and Decoder implementations
    # For example purposes, we'll just ensure that there are some parameters
    num_params = sum(p.numel() for p in transformer.parameters())

    # TODO: Calculate the amount of parameters that the model should ideally have
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
    Test that the Transformer's forward pass produces valid output values (e.g., probabilities).

    Parameters:
    - batch_size (int): Number of samples in a batch.
    - seq_len (int): Length of the input sequences.
    - d_model (int): Dimension of the model (embedding size).
    - n (int): Number of encoder and decoder layers.
    - number_of_heads (int): Number of attention heads.
    - vocab_size (int): Size of the target vocabulary.
    """
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

    # Perform a forward pass
    output = transformer(encoder_x, decoder_x, apply_softmax=True)

    # Since the output applies a softmax, all values should be between 0 and 1
    assert torch.all(output >= 0) and torch.all(
        output <= 1
    ), "Output values are not in [0, 1]"

    # Additionally, check that each vector along the softmax dimension sums to 1
    sums = output.sum(dim=2)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-6
    ), "Softmax outputs do not sum to 1"

    # Verify the output shape matches (batch_size, seq_len, vocab_size)
    assert output.size() == (batch_size, seq_len, vocab_size), (
        f"Expected output shape {(batch_size, seq_len, vocab_size)}, "
        f"but got {output.size()}"
    )
