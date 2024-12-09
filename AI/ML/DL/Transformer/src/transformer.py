import math
import torch
import numpy as np


def softmax(x: torch.Tensor, d: int):
    # return torch.nn.Softmax(dim=d)(x)
    return torch.nn.functional.softmax(x, dim=d)


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        hidden_size: list[tuple[int]] = None,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        """
        notes:
            - default dimensions are set according to paper

        """
        if hidden_size is None:
            hidden_size = [(d_model, 2048), (2048, d_model)]

        assert len(hidden_size) > 0, "hidden_size must be greater than 0"
        assert (
            hidden_size[0][0] == d_model and hidden_size[-1][1] == d_model
        ), "input and output dimensions must equal d_model"

        super().__init__()

        layers = []
        n = len(hidden_size)
        for i in range(n):
            layers.append(torch.nn.Linear(*hidden_size[i]))

            # No activation after the final layer
            if i < n - 1:
                layers.append(activation)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pad_mask: torch.Tensor,
    mask: torch.Tensor = None,
):
    """
    input:
        q: a torch tensor of size: (batch_size*h, seq_len, d_k)
        k: a torch tensor of size: (batch_size*h, seq_len, d_k)
        v: a torch tensor of size: (batch_size*h, seq_len, d_v)
        pad_mask: a torch tensor of size: (batch_size*h, seq_len, seq_len)
        mask: a torch tensor of size: (seq_len, seq_len)

    output:
        A torch tensor of size: (batch_size*h, seq_len, d_v)

    """
    x = torch.matmul(q, torch.transpose(k, 1, 2))

    # Scale x by sqrt(d_k)
    x = x / math.sqrt(q.size(2))

    if mask is not None:

        # Broadcasting will match the size at dim=0
        mask = mask.unsqueeze(0)

        # Apply mask via element-wise addition
        x = x + mask

    x = x + pad_mask

    return torch.matmul(softmax(x, d=2), v)


class MultiHead(torch.nn.Module):
    def __init__(self, h: int = 8, d_model: int = 512, mask: torch.Tensor = None):
        """
        input:
            h: number of heads
            d_model: model dimensions, i.e. embedding size
            mask: a boolean to apply masked multi-head attention
        notes:
            - default dimensions are set according to the paper

        """
        assert d_model % h == 0, "d_model must be divisible by h"
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = self.d_k
        self.h = h
        self.mask = mask

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_mask: torch.Tensor
    ):
        """
        input:
            q: a torch tensor of size: (batch_size, seq_len, d_model)
            k: a torch tensor of size: (batch_size, seq_len, d_model)
            v: a torch tensor of size: (batch_size, seq_len, d_model)
            pad_mask: a torch tensor of size: (batch_size, seq_len, seq_len)

        output:
            A torch tensor of size: (batch_size, seq_len, d_model)

        notes:
            - d_model is essentially embedding dimensions
        """
        batch_size, seq_len, _ = q.size()
        q_h = self.w_q(q)
        k_h = self.w_k(k)
        v_h = self.w_v(v)

        # Splitting q, k and v tensors in to h heads
        q_h = q_h.reshape(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        k_h = k_h.reshape(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        v_h = v_h.reshape(batch_size, seq_len, self.h, self.d_v).transpose(1, 2)

        # Combine heads for parrallel computation
        q_h = q_h.reshape(batch_size * self.h, seq_len, self.d_k)
        k_h = k_h.reshape(batch_size * self.h, seq_len, self.d_k)
        v_h = v_h.reshape(batch_size * self.h, seq_len, self.d_v)

        # Apply attention
        scores = attention(q_h, k_h, v_h, pad_mask, mask=self.mask)

        # Seperate heads
        scores = scores.reshape(batch_size, self.h, seq_len, self.d_v).transpose(1, 2)

        # Concat h heads (Concat(head1, ..., headh))
        scores = scores.reshape(batch_size, seq_len, self.h * self.d_v)

        return self.w_o(scores)


# TODO: can change the definition so that if dropout called with
# p=None then can just terminate. This would make the Encoder and
# Decoder forward passes more concise
def dropout(x: torch.Tensor, p: float = 0.1):
    assert 0 <= p <= 1, "p must be a probability"

    if p == 1:
        # All elements are dropped; just return zeros.
        return torch.zeros_like(x)

    # Create a tensor with the same shape as x
    # and set all is values to 1 - p
    mask = torch.full_like(x, 1 - p)

    # Will sample the entries from the bernoulli distribution.
    # The i'th entry of the output tensor will draw a value 1 according
    # to the i'th probability given the input tensor.
    mask = torch.bernoulli(mask).to(x.device)

    # Apply dropout via element-wise multiplication.
    x = x * mask

    # Apply inverted scaling
    return x * (1 / (1 - p))


class Encoder(torch.nn.Module):
    def __init__(self, number_of_heads: int = 8, d_model: int = 512):
        """
        notes:
            - default parameter values are based on the paper
        """
        assert (
            d_model % number_of_heads == 0
        ), "d_model must be divisible by number_of_heads"
        super().__init__()
        self.multi_head_attention = MultiHead(h=number_of_heads, d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model)
        self.d_model = d_model
        self.layer_norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(d_model) for _ in range(2)
        )

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor, dropout_p=None
    ) -> torch.Tensor:
        """
        input:
            x: a torch tensor of size: (batch_size, seq_len, d_model)
        output:
            a torch tensor of size: (batch_size, seq_len, d_model)
        """

        x = self.layer_norms[0](
            x
            + (
                self.multi_head_attention(x, x, x, pad_mask)
                if dropout_p is None
                else dropout(self.multi_head_attention(x, x, x, pad_mask), p=dropout_p)
            )
        )

        return self.layer_norms[1](
            x
            + (
                self.feed_forward(x)
                if dropout_p is None
                else dropout(self.feed_forward(x), p=dropout_p)
            )
        )


class Decoder(torch.nn.Module):
    def __init__(
        self,
        attention_mask: torch.Tensor,
        number_of_heads: int = 8,
        d_model: int = 512,
    ):
        """
        notes:
            - default parameter values are based on the paper
            - in the previous implementation we were using the same multi head
            module and inidicating the apply mask on the forward pass. That's an
            outregous mistake. The same parameters are being used in that case
            only when applying attention part of the sequence was getting masked.
        """
        assert (
            d_model % number_of_heads == 0
        ), "d_model must be divisible by number_of_heads"
        super().__init__()
        self.multi_head_attention = MultiHead(h=number_of_heads, d_model=d_model)
        self.masked_multi_head_attention = MultiHead(
            h=number_of_heads, d_model=d_model, mask=attention_mask
        )
        self.feed_forward = FeedForward(d_model=d_model)
        self.d_model = d_model
        self.layer_norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(d_model) for _ in range(3)
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_x: torch.Tensor,
        decoder_pad_mask: torch.Tensor,
        encoder_pad_mask: torch.Tensor,
        dropout_p=None,
    ) -> torch.Tensor:
        """
        inputs:
            x: a torch tensor of size: (batch_size, seq_len, d_model)
            encoder_x: a torch tensor of size: (batch_size, seq_len, d_model)
        output:
            a torch tensor of size: (batch_size, seq_len, d_model)
        """
        x = self.layer_norms[0](
            x
            + (
                self.masked_multi_head_attention(x, x, x, decoder_pad_mask)
                if dropout_p is None
                else dropout(
                    self.masked_multi_head_attention(x, x, x, decoder_pad_mask),
                    p=dropout_p,
                )
            )
        )

        # Cross-Attention
        # Here encoder_pad_mask must be used because
        x = self.layer_norms[1](
            x
            + (
                self.multi_head_attention(x, encoder_x, encoder_x, encoder_pad_mask)
                if dropout_p is None
                else dropout(
                    self.multi_head_attention(
                        x, encoder_x, encoder_x, encoder_pad_mask
                    ),
                    p=dropout_p,
                )
            )
        )

        return self.layer_norms[2](
            x
            + (
                self.feed_forward(x)
                if dropout_p is None
                else dropout(self.feed_forward(x), p=dropout_p)
            )
        )


def positional_encodings(seq_len: int, d_model: int) -> np.ndarray:
    pos, i = np.indices((seq_len, d_model))
    return np.where(
        i % 2 == 0,
        np.sin(pos / np.power(10000, (2 * i / d_model))),
        np.cos(pos / np.power(10000, (2 * i / d_model))),
    )


def get_attention_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Here we will be creating a triangular matrix where
    all the upper triangle (above the diagonal) is set to -oo.

    output:
        A torch tensor of size: (seq_len, seq_len)
    """

    mask_x = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)

    return torch.triu(mask_x, diagonal=1)

    # Add a new batch_size dimension and expand it to
    # match batch_size
    # -1 means keep the size at that dimension
    # However its also documented here:
    # https://stackoverflow.com/questions/65900110/does-pytorch-broadcast-consume-less-memory-than-expand
    # that expand does not also consume extra memory
    # mask = mask.unsqueeze(0).expand(batch_size, -1, -1)


def get_pad_mask(x: torch.Tensor, h, batch_size, seq_len) -> torch.Tensor:
    """
    inputs:
        x: a torch tensor of size: (batch_size, seq_len)
    """
    # 0 must be the padding token id
    # Here torch.where(encoder_x == 0, float("-inf"), 0)
    #    .unsqueeze(2)
    #    .expand(-1, -1, self.seq-len)
    # should also mathematically produce the same shape
    # but typically the keys gets masked so the current approach
    # aligns better with the paper
    assert x.size() == (
        batch_size,
        seq_len,
    ), f"x must have size: {(batch_size, seq_len)}"

    return (
        torch.where(x == 0, float("-inf"), 0)
        .unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, h, seq_len, -1)
        .reshape(batch_size * h, seq_len, seq_len)
    )


class Transformer(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int = 512,
        n: int = 6,
        number_of_heads: int = 8,
        input_vocab_size: int = 37000,
        output_vocab_size: int = 37000,
        dtype: torch.dtype = torch.float32,
    ):
        """
        inputs:
            n: the number of encoder and decoder stacks
            d_model: model dimensions, i.e. embedding dimensions
            number_of_heads: number of heads using in multi-head attention
            input_vocab_size: size of the vocabulary for the input structure
            output_vocab_size: size of the vocabulary for the output structure

        notes:
            - I'm using a fixed seq_len for both the input and the output. That could
            be adjusted to make it varied and more flexible.

        """
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n = n
        self.d_model = d_model
        self.number_of_heads = number_of_heads
        self.vocab_size = output_vocab_size
        self.dtype = dtype
        self.register_buffer(
            "pe",
            torch.from_numpy(positional_encodings(seq_len, d_model))
            .to(dtype)
            .unsqueeze(0),
        )
        self.register_buffer("attention_mask", get_attention_mask(seq_len, dtype=dtype))
        self.linear = torch.nn.Linear(d_model, output_vocab_size)

        self.input_embedding = torch.nn.Embedding(
            num_embeddings=input_vocab_size, embedding_dim=d_model
        )
        self.output_embedding = torch.nn.Embedding(
            num_embeddings=output_vocab_size, embedding_dim=d_model
        )

        # Using ModuleList is crucial here instead of python list
        # If python list is used, then model.parameters() will not
        # return the paremeters of layers inside.
        self.encoders = torch.nn.ModuleList(
            Encoder(number_of_heads=number_of_heads, d_model=d_model)
            for _ in range(self.n)
        )

        self.decoders = torch.nn.ModuleList(
            Decoder(
                attention_mask=self.attention_mask,
                number_of_heads=number_of_heads,
                d_model=d_model,
            )
            for _ in range(self.n)
        )

    def forward(
        self,
        encoder_x: torch.Tensor,
        decoder_x: torch.Tensor,
        apply_softmax: bool = False,
    ):
        """
        inputs:
            encoder_x: a torch tensor of size: (batch_size, seq_len)
            decoder_x: a torch tensor of size: (batch_size, seq_len)
            apply_softmax: a boolean.
                - Most torch loss functions expect logits instead of porbabilities.
                So make sure that the loss function does not normalize inputs and
                expects probabilities before setting this to True.

        outputs:
            a probability distribution over the vocabulary
        """
        assert (
            encoder_x.size() == decoder_x.size() == (self.batch_size, self.seq_len)
        ), f"encoder_x and decoder_x must both have the size: ({self.batch_size}, {self.seq_len})"

        assert torch.any(encoder_x) and torch.any(decoder_x), (
            "empty examples are not allowed. There could be some additional reasons "
            "for not to allow them but simple example why is that softmax is not defined "
            "over an empty sequence"
        )

        encoder_x_pad_mask = get_pad_mask(
            encoder_x,
            h=self.number_of_heads,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
        )
        decoder_x_pad_mask = get_pad_mask(
            decoder_x,
            h=self.number_of_heads,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
        )

        # In the paper it is mentioned that they scale the embedding weights by math.sqrt(self.d_model)
        # see the end of section 3.4 for more detail
        encoder_x = self.input_embedding(encoder_x) * math.sqrt(self.d_model) + self.pe
        decoder_x = self.output_embedding(decoder_x) * math.sqrt(self.d_model) + self.pe

        for encoder in self.encoders:
            encoder_x = encoder(encoder_x, encoder_x_pad_mask)

        for decoder in self.decoders:
            decoder_x = decoder(
                decoder_x,
                encoder_x,
                decoder_pad_mask=decoder_x_pad_mask,
                encoder_pad_mask=encoder_x_pad_mask,
            )

        return (
            softmax(self.linear(decoder_x), d=2)
            if apply_softmax
            else self.linear(decoder_x)
        )
