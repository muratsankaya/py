import pytest
import numpy as np
from tests.helper_functions import print_green
from src.transformer_v1 import positional_encodings


@pytest.mark.parametrize("seq_len, d_model", [(4, 4), (32, 128), (128, 512)])
def test_positional_encoding(seq_len, d_model):
    x = positional_encodings(seq_len, d_model)
    assert isinstance(x, np.ndarray), "positional encodings must return np.ndarray"
    assert x.shape == (
        seq_len,
        d_model,
    ), "the output size should match with the input parameters"

    # Choose a random position pos, and dimension i
    # make sure that its encoded correctly
    for _ in range(5):  # for sanity repeat 5 times
        pos, i = np.random.randint(seq_len), np.random.randint(d_model)
        encoding = pos / np.power(10000, 2 * i / d_model)
        encoding = np.sin(encoding) if i % 2 == 0 else np.cos(encoding)
        assert np.isclose(encoding, x[pos, i]), "there is a mistake in the encoding"

    print_green("PE tests pass")
