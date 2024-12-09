import torch
import torch.nn.functional as F
from helper_functions import print_green
from src.transformer import attention, get_attention_mask, get_pad_mask


def test_attention_without_mask():
    """
    Test the attention function without any masks (no causal mask, no pad mask).
    """
    batch_size = 1
    seq_len = 3
    d_k = 1
    d_v = 1
    h = 1

    # Define deterministic Q, K, V tensors with no zero (no padding)
    Q = torch.tensor([[[1.0], [1.0], [1.0]]])  # Shape: (1, 3, 1)
    K = torch.tensor([[[1.0], [1.0], [1.0]]])  # Shape: (1, 3, 1)
    V = torch.tensor([[[1.0], [2.0], [3.0]]])  # Shape: (1, 3, 1)

    # Compute pad_mask using get_pad_mask:
    # Since there are no zeros, pad_mask should be all zeros
    pad_mask = get_pad_mask(Q.squeeze(-1), h=h, batch_size=batch_size, seq_len=seq_len)

    # Expected attention output
    expected_output = torch.tensor([[[2.0], [2.0], [2.0]]])

    # Compute attention without causal mask
    output = attention(Q, K, V, pad_mask=pad_mask, mask=None)

    # Check output shape and values
    assert output.size() == (
        batch_size,
        seq_len,
        d_v,
    ), f"Expected {(batch_size, seq_len, d_v)}, got {output.size()}"
    assert torch.allclose(
        output, expected_output
    ), f"Expected {expected_output}, got {output}"
    assert isinstance(output, torch.Tensor), "Output is not a PyTorch tensor"

    print_green("test_attention_without_mask passed.")


def test_attention_with_mask():
    """
    Test the attention function with a causal mask.
    """
    batch_size = 1
    seq_len = 3
    d_k = 1
    d_v = 1
    h = 1

    # No padding scenario again
    Q = torch.tensor([[[1.0], [1.0], [1.0]]])  # Shape: (1, 3, 1)
    K = Q.clone()
    V = torch.tensor([[[1.0], [2.0], [3.0]]])  # Shape: (1, 3, 1)

    # pad_mask with no zeros in Q
    pad_mask = get_pad_mask(Q.squeeze(-1), h=h, batch_size=batch_size, seq_len=seq_len)

    # Expected output under causal mask
    expected_output = torch.tensor([[[1.0], [1.5], [2.0]]])

    # Compute attention with causal mask
    output = attention(
        Q, K, V, pad_mask=pad_mask, mask=get_attention_mask(seq_len, dtype=Q.dtype)
    )

    # Check output
    assert output.size() == (
        batch_size,
        seq_len,
        d_v,
    ), f"Expected {(batch_size, seq_len, d_v)}, got {output.size()}"
    assert torch.allclose(
        output, expected_output
    ), f"Expected {expected_output}, got {output}"
    assert isinstance(output, torch.Tensor), "Output is not a PyTorch tensor"

    print_green("test_attention_with_mask passed.")


def test_attention_batch():
    """
    Test the attention function with batching and a causal mask.
    """
    batch_size = 2
    seq_len = 3
    d_k = 1
    d_v = 1
    h = 1

    Q = torch.tensor([[[1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]]])  # Shape: (2, 3, 1)
    K = Q.clone()
    V = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])  # Shape: (2, 3, 1)

    # No zeros => no padding
    pad_mask = get_pad_mask(Q.squeeze(-1), h=h, batch_size=batch_size, seq_len=seq_len)

    # Expected output with causal mask
    expected_output = torch.tensor([[[1.0], [1.5], [2.0]], [[4.0], [4.5], [5.0]]])

    # Compute attention with causal mask
    output = attention(
        Q, K, V, pad_mask=pad_mask, mask=get_attention_mask(seq_len, Q.dtype)
    )

    assert output.size() == (
        batch_size,
        seq_len,
        d_v,
    ), f"Expected {(batch_size, seq_len, d_v)}, got {output.size()}"
    assert torch.allclose(
        output, expected_output
    ), f"Expected {expected_output}, got {output}"
    assert isinstance(output, torch.Tensor), "Output is not a PyTorch tensor"

    print_green("test_attention_batch passed.")


def test_attention_with_pad_mask():
    batch_size = 3
    seq_len = 4
    d_k = 1
    d_v = 1
    h = 1

    # Let's define:
    # For batch 0: tokens = [T1, T2, PAD, PAD]
    # For batch 1: tokens = [T3, T4, T5, T6] (no padding)
    # We'll create Q, K, V identical. Values for non-pad: distinct positive values.
    # Values for PAD: we can keep them as zeros or distinct but they should be masked out anyway.

    Q = torch.tensor(
        [
            [[1.0], [2.0], [0.0], [0.0]],  # Batch 0
            [[3.0], [4.0], [5.0], [6.0]],  # Batch 1, no padding
            [
                [5.0],
                [0.0],
                [0.0],
                [0.0],
            ],  # Try the most extreme case when there is only a single token in the sequence
        ]
    )  # shape: (3,4,1)

    K = Q.clone()
    V = Q.clone()

    # Construct a pad_mask:
    # For batch 0: positions 2 and 3 are pad tokens (e.g. token_id=0)
    # For batch 1: no padding
    # pad_mask should have shape (batch_size*h, seq_len, seq_len), but here we'll test it directly.

    # Now we use get_pad_mask to handle these zeros:
    pad_mask = get_pad_mask(Q.squeeze(-1), h=h, batch_size=batch_size, seq_len=seq_len)

    # No causal mask here, just pad mask
    # Compute attention
    scores = attention(
        Q.reshape(batch_size * h, seq_len, d_k),
        K.reshape(batch_size * h, seq_len, d_k),
        V.reshape(batch_size * h, seq_len, d_v),
        pad_mask=pad_mask,
        mask=None,
    )

    # Reshape back to (batch_size, seq_len, d_v)
    # scores = scores.reshape(batch_size, seq_len, d_v)

    # Let's reason about the expected output:
    # For Batch 0:
    # Q and K are [1,2,0,0], so for queries at positions 0 and 1, we have two non-pad tokens (positions 0 and 1) and two pad tokens (positions 2 and 3).
    # The attention should ignore positions 2 and 3.
    # If we consider just positions 0 and 1:
    # For query at pos 0: K = [1,2,0,0], after softmax ignoring pads, we should get attention roughly split between tokens 1 and 2 (since no difference in QK?),
    # but let's be exact:
    #
    # scores = Q*K^T / sqrt(d_k)
    # For batch 0, query pos 0 (Q=1):
    # scores = [1*1, 1*2, 1*0, 1*0] = [1,2,-inf,-inf]
    # softmax([1,2,-inf,-inf]) ~ [0.269, 0.731, 0, 0]
    # Weighted sum on V: V0=1, V1=2 => 1*0.269 + 2*0.731 = 1.731 ~ 1.73
    #
    # For query pos 1 (Q=2):
    # scores = [2*1,2*2,2*0,2*0]=[2,4,-inf,-inf]
    # softmax([2,4,-inf,-inf]) ~ [0.119,0.881,0,0]
    # Weighted sum: V0=1,V1=2 => 1*0.119 + 2*0.881=1.881 ~ 1.88
    #
    # For query pos 2 (Q=0, pad query):
    # scores = [0*1,0*2,0*0,0*0]=[0,0,-inf,-inf]? Actually, Q is zero. If Q=0,
    # It's still a query we need an output for.
    # The pad_mask affects keys, not queries. That means queries at pos 2 see no valid keys:
    # Actually, we might need to rethink: If Q=0 (a pad query), there's no reason it should attend anywhere.
    # The attention would still happen, but pad_mask on keys sets positions 2 and 3 to -inf. Positions 0 and 1 are not pad, so keys 0 and 1 are valid:
    # scores = [0*1,0*2,0*0,0*0] = [0,0,-inf,-inf]
    # softmax([0,0,-inf,-inf]) = [0.5,0.5,0,0]
    # Weighted sum: 1*0.5+2*0.5=1.5
    #
    # Query pos 3 (Q=0):
    # Same logic as pos 2:
    # scores = [0,0,-inf,-inf]
    # softmax = [0.5,0.5,0,0]
    # Weighted sum = 1.5
    #
    # For Batch 1: no pads, so attention is uniform:
    # Q,K,V: [3,4,5,6],
    # Each query sees all keys:
    # Query pos 0:
    # scores ~ softmax of something symmetrical, but Q=3 with K=3,4,5,6
    # Actually let's not get too complicated, the key point is that no pad_mask affects batch 1,
    # so it should just work like regular attention.
    #
    # For simplicity, just run the test and confirm output doesn't produce NaNs and that non-pad queries do not get attention on pad keys.

    # Basic sanity check: no NaNs and finite:
    assert torch.isfinite(scores).all(), "Output contains non-finite values."

    # Check that padded positions in batch 0 do not get influence from pad keys:
    # Positions 2 and 3 queries ended up averaging only the non-pad keys (0 and 1),
    # This is correct given the mask we added.

    assert (
        scores[0, 2] == scores[0, 3] == 1.5
    ), "Attention scores on batch 1 q{2,3} should equal 1.5 see comments for calculation"

    assert torch.all(
        scores[2, 1:] == 5.0
    ), "Only token in batch 2 has embedding value of 5, when attention applied should be 5 so 5*1 + 0*0 ... = 5"

    assert scores[2, 0] == 5.0, (
        "Since there is only one token that attention is applied to it also should be eq. to the q value"
        " so that should also be 5.0"
    )

    print_green("test_attention_with_pad_mask passed.")


def test_attention_multiple_heads_pad_and_causal():
    batch_size = 2
    seq_len = 5
    h = 2
    d_k = 1
    d_v = 1

    # Construct sequences:
    # Batch 0: [1,2,0,0,3] with pads at pos2,3
    # Batch 1: [4,5,6,7,8] no pads
    tokens = torch.tensor(
        [[1, 2, 0, 0, 3], [4, 5, 6, 7, 8]]  # Batch 0  # Batch 1
    )  # shape: (2,5)

    # Q,K:
    # For Batch 0: Q,K = 1 at non-pad (pos0,1,4), Q,K=0 at pad (pos2,3)
    batch0_QK = torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]])
    # For Batch 1: all non-pad, Q,K=1 for all positions
    batch1_QK = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]])

    Q = torch.cat([batch0_QK, batch0_QK, batch1_QK, batch1_QK], dim=0).reshape(
        batch_size * h, seq_len, d_k
    )
    K = Q.clone()

    # V:
    # Batch 0 values: [10,20,0,0,30]
    batch0_V = torch.tensor([[10.0], [20.0], [0.0], [0.0], [30.0]])
    # Batch 1 values: [4,5,6,7,8]
    batch1_V = torch.tensor([[4.0], [5.0], [6.0], [7.0], [8.0]])

    V = torch.cat([batch0_V, batch0_V, batch1_V, batch1_V], dim=0).reshape(
        batch_size * h, seq_len, d_v
    )

    # Masks
    pad_mask = get_pad_mask(tokens, h=h, batch_size=batch_size, seq_len=seq_len)
    causal_mask = get_attention_mask(seq_len, dtype=Q.dtype)

    # Compute attention
    scores = attention(Q, K, V, pad_mask=pad_mask, mask=causal_mask)

    # Check shapes and finiteness
    assert scores.shape == (
        batch_size * h,
        seq_len,
        d_v,
    ), f"Expected {(batch_size*h, seq_len, d_v)}, got {scores.shape}"
    assert torch.isfinite(scores).all(), "Output contains non-finite values."

    # --- Verify Causal Mask at an Early Position ---

    # Consider Batch 1 (indices 2 and 3), Position 1:
    # Without a causal mask, position 1 could attend [0,1,2,3,4].
    # With a causal mask, position 1 attends only [0,1].
    # Batch 1 tokens: [4,5,6,7,8]
    # If only [0,1] (that is 4 and 5) are attended, average = (4+5)/2 = 4.5.
    # Check head 0 of batch 1 (index=2) at position 1:
    assert torch.allclose(
        scores[2, 1], torch.tensor([4.5])
    ), "Batch 1 head 0 pos 1 should average [4,5] = 4.5 due to causal mask"
    # Check head 1 of batch 1 (index=3) at position 1:
    assert torch.allclose(
        scores[3, 1], torch.tensor([4.5])
    ), "Batch 1 head 1 pos 1 should also be 4.5"

    # --- Verify Pad Mask + Causal Mask at a Later Position ---
    # For Batch 0 (indices 0 and 1), position 4:
    # Non-pad keys: positions 0=10,1=20,4=30. Positions 2,3 are pad.
    # With causal mask, position 4 can attend [0..4].
    # But pad_mask removes pos2,3. So effectively attends [10,20,30].
    # Average = (10+20+30)/3=20.
    assert torch.allclose(
        scores[0, 4], torch.tensor([20.0])
    ), "Batch 0 head 0 pos 4 expected ~20.0"
    assert torch.allclose(
        scores[1, 4], torch.tensor([20.0])
    ), "Batch 0 head 1 pos 4 expected ~20.0"

    # For Batch 1 (indices 2 and 3), position 4:
    # Positions [0..4] = [4,5,6,7,8]
    # All attended (no pad), average = 30/5=6.0
    assert torch.allclose(
        scores[2, 4], torch.tensor([6.0])
    ), "Batch 1 head 0 pos 4 expected 6.0"
    assert torch.allclose(
        scores[3, 4], torch.tensor([6.0])
    ), "Batch 1 head 1 pos 4 expected 6.0"

    print_green("test_attention_multiple_heads_pad_and_causal passed.")
