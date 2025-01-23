import math
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

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
        return torch.zeros_like(x).to(device).to(dtype)

    # Create a tensor with the same shape as x
    # and set all is values to 1 - p
    mask = torch.full_like(x, 1 - p).to(device).to(dtype)

    # Will sample the entries from the bernoulli distribution.
    # The i'th entry of the output tensor will draw a value 1 according
    # to the i'th probability given the input tensor.
    mask = torch.bernoulli(mask).to(device).to(dtype)

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

    mask_x = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype).to(device).to(dtype)

    return torch.triu(mask_x, diagonal=1).to(device).to(dtype)

    # Add a new batch_size dimension and expand it to
    # match batch_size
    # -1 means keep the size at that dimension
    # However its also documented here:
    # https://stackoverflow.com/questions/65900110/does-pytorch-broadcast-consume-less-memory-than-expand
    # that expand does not also consume extra memory
    # mask = mask.unsqueeze(0).expand(batch_size, -1, -1)


def get_pad_mask(
    x: torch.Tensor, h: int, batch_size: int, seq_len: int, pad_token_id: int = 0
) -> torch.Tensor:
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
        torch.where(x == pad_token_id, float("-inf"), 0)
        .unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, h, seq_len, -1)
        .reshape(batch_size * h, seq_len, seq_len)
        .to(device)
        .to(dtype)
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
        dropout_p: float = None,
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
            .unsqueeze(0)
            .to(device)
        )
        self.register_buffer("attention_mask", get_attention_mask(seq_len, dtype=dtype))
        self.linear = torch.nn.Linear(d_model, output_vocab_size)
        assert (
            dropout_p is None or 0 <= dropout_p <= 1
        ), "p_dropout must be a value between 0 and 1"
        self.dropout_p = dropout_p

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
        pad_token_id: int = 0,
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
            a probability distribution over the vocabulary if apply_softmax is true else
            it outputs logits
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
            pad_token_id=pad_token_id,
        )
        decoder_x_pad_mask = get_pad_mask(
            decoder_x,
            h=self.number_of_heads,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            pad_token_id=pad_token_id,
        )

        # In the paper it is mentioned that they scale the embedding weights by math.sqrt(self.d_model)
        # see the end of section 3.4 for more detail
        encoder_x = self.input_embedding(encoder_x) * math.sqrt(self.d_model) + self.pe
        decoder_x = self.output_embedding(decoder_x) * math.sqrt(self.d_model) + self.pe

        for encoder in self.encoders:
            encoder_x = encoder(encoder_x, encoder_x_pad_mask, dropout_p=self.dropout_p)

        for decoder in self.decoders:
            decoder_x = decoder(
                decoder_x,
                encoder_x,
                decoder_pad_mask=decoder_x_pad_mask,
                encoder_pad_mask=encoder_x_pad_mask,
                dropout_p=self.dropout_p,
            )

        return (
            softmax(self.linear(decoder_x), d=2)
            if apply_softmax
            else self.linear(decoder_x)
        )
        
        
        
        
        
        
from datasets import load_dataset

wmt14_train = load_dataset("wmt14", "de-en", split="train[:10000]")
wmt14_test = load_dataset("wmt14", "de-en", split="test[:3000]")
wmt14_validation = load_dataset("wmt14", "de-en", split="validation[:7000]")

# # Display available splits
# print(wmt14)

# # Display column names and data types for the 'train' split
# print(wmt14["train"].features)

# # Display the first 5 rows of the 'train' split

print("\nprinting 2 examples from the train dataset:")
for i in range(2):
    print(wmt14_train[i])

print("\nprinting 2 examples from the test dataset:")
for i in range(2):
    print(wmt14_test[i])

print("\nprinting 2 examples from the validation dataset")
for i in range(2):
    print(wmt14_validation[i])





from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Pre-trained English-German tokenizer/model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# print("Tokenizer pad token:", tokenizer.pad_token)  # prints: '<pad>'
# print("Tokenizer pad token id:", tokenizer.pad_token_id)  # prints: '58100'


# Determine max length of tokenized sequences before padding
# def length_check_function(examples):
#     # Temporarily no truncation or padding
#     src_lengths, tgt_lengths = [], []
#     for translation in examples["translation"]:
#         src_lengths.append(len(tokenizer.tokenize(translation["de"])))
#         tgt_lengths.append(len(tokenizer.tokenize(translation["en"])))

#     return {"src_length": src_lengths, "tgt_length": tgt_lengths}


# lengths = wmt14["train"].map(length_check_function, batched=True)
# max_source_length = max(lengths["src_length"])
# max_target_length = max(lengths["tgt_length"])
# max_seq_len = max(max_source_length, max_target_length)
# print("Max sequence length:", max_seq_len)  # prints: '13,614'

max_seq_len = 32


if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({"bos_token": "<s>"})
    # If a model is already loaded, you may need:
    model.resize_token_embeddings(len(tokenizer))


def preprocess_function(examples):
    # Tokenize source (German)
    model_inputs = tokenizer(
        [translation["de"] for translation in examples["translation"]],
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
    )

    # Tokenize target (English)
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(
            [translation["en"] for translation in examples["translation"]],
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
        )

    input_ids = tokenized_targets["input_ids"]

    # Ensure bos_token_id is defined (after adding bos_token if needed)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # Create decoder_input_ids by prepending BOS and removing the last token
    # Example:
    # original input_ids: [w1, w2, w3, eos]
    # decoder_input_ids:  [bos_id, w1, w2, w3]
    # labels:             [w1, w2, w3, eos]
    decoder_input_ids = [
        [bos_id] + [seq_id for seq_id in seq if seq_id != eos_id] for seq in input_ids
    ]

    # When indicies are set to -100 they are ignored in loss computation of cross entropy
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    labels = [
        [(seq_id if seq_id != pad_id else -100) for seq_id in seq] for seq in input_ids
    ]

    # Update model_inputs
    model_inputs["labels"] = labels
    model_inputs["decoder_input_ids"] = decoder_input_ids
    model_inputs["decoder_attention_mask"] = tokenized_targets["attention_mask"]

    return model_inputs


def preprocess_dataset(dataset, num_of_examples):
    """
    Preprocesses a given dataset.

    Args:
      dataset: The dataset to preprocess.
      preprocess_function: The preprocessing function to apply.
      tokenizer: The tokenizer to use for decoding.

    Returns:
      A preprocessed dataset.
    """

    # Select a subset of the dataset
    dataset_subset = dataset.select(range(num_of_examples))

    # Apply the preprocessing function
    tokenized_dataset_subset = dataset_subset.map(
        preprocess_function, batched=True, remove_columns=dataset.column_names
    )

    # Convert to torch for consistency
    tokenized_dataset_subset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "decoder_input_ids",
            "labels",
            "attention_mask",
            "decoder_attention_mask",
        ],
    )

    # Print some examples to visually inspect them
    for i in range(3):
        print("Example", i)
        print("Input IDs:", tokenized_dataset_subset[i]["input_ids"])
        print("Decoder Input IDs:", tokenized_dataset_subset[i]["decoder_input_ids"])
        print("Labels:", tokenized_dataset_subset[i]["labels"])
        print("Encoder Attention Mask:", tokenized_dataset_subset[i]["attention_mask"])
        print(
            "Decoder Attention Mask:",
            tokenized_dataset_subset[i]["decoder_attention_mask"],
        )

        print(
            f"decoded_input size: {tokenized_dataset_subset[i]['input_ids'].size()}",
            "Decoded input:",
            tokenizer.decode(tokenized_dataset_subset[i]["input_ids"]).replace(
                " <pad>", ""
            ),
        )
        print(
            f"decoded_decoder_input size: {tokenized_dataset_subset[i]['decoder_input_ids'].size()}",
            "Decoded decoder input:",
            tokenizer.decode(tokenized_dataset_subset[i]["decoder_input_ids"])
            .replace(" <pad>", "")
            .replace("<pad>", ""),
        )
        print()

    return tokenized_dataset_subset


print("Printing preprocessed train dataset:")
wmt14_train_subset = preprocess_dataset(wmt14_train, num_of_examples=5000)
print(f"train dataset size: {len(wmt14_train_subset)}")

print("Printing preprocessed test dataset:")
wmt14_test_subset = preprocess_dataset(wmt14_test, num_of_examples=3000)
print(f"test dataset size: {len(wmt14_test_subset)}")

print("Printing preprocessed validation dataset:")
wmt14_validation_subset = preprocess_dataset(wmt14_validation, num_of_examples=2000)
print(f"validation dataset size: {len(wmt14_validation_subset)}")







import os

def save(filename, **kwargs):
    """
    Save a pytorch object to file
    See: https://pytorch.org/tutorials/beginner/saving_loading_models.html

    Arguments:
        filename: the file in which to save the object

    Possible keyword arguments (kwargs):
        epoch: the epoch so far if training
        model_state_dict: a model's state
        opt_state_dict: a optimizer's state, if training
    """

    msg = f"{filename} exists: delete it first to replace it."
    assert not os.path.exists(filename), msg
    torch.save(kwargs, filename)


def load(filename):
    """
    Load a pytorch object from a given filename
    See: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    You shouldn't need to edit this function.

    Arguments:
        filename: the file from which to load the object
    """

    return torch.load(filename)




import time



class Trainer:
    def __init__(self, optimizer, model, loss_func, scheduler=None, **kwargs):
        """
        Initialize the optimizer for the model, using any necessary kwargs
        Save the model and loss function for later calculation
        You shouldn't need to edit this function.
        """

        self.optimizer: torch.optim.Optimizer = optimizer(model.parameters(), **kwargs)
        self.scheduler = scheduler["scheduler"](
            self.optimizer, lr_lambda=scheduler["lr_lambda"]
        )
        self.model: torch.nn.Module = model
        self.loss_func = loss_func

        self.epoch = 0
        self.start_time = None

    def run_one_batch(self, encder_x, decoder_x, y, train=True, pad_token_id=0):
        """
        Run self.model on one batch of data, using `self.loss_func` to
            compute the model's loss.

        If train=True (the default), you should use `self.optimizer`
            to update the parameters of `self.model`.

        You should also call `self.optimizer.zero_grad()`; see
            https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            for a guide as to when to do that.

        Args
            enocder_x: the batch's enocder input
            decoder_x: the batch's decoder input
            y: the batch's target

        Returns
            loss: the model's loss on this batch
        """

        if train:
            self.optimizer.zero_grad()

        outputs = self.model(
            encder_x.to(device), decoder_x.to(device), pad_token_id=pad_token_id
        )
        loss = self.loss_func(outputs.transpose(1, 2), y.to(device))

        if train:
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return loss.detach().cpu().numpy()

    def run_one_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        train=True,
        verbose=False,
        pad_token_id=0,
    ):
        """
        Train one epoch, a batch at a time, using self.run_one_batch
        You shouldn't need to edit this function.

        Args:
            data_loader: a torch.utils.data.DataLoader with our dataset
            stats: an optional dict of information to print out

        Returns:
            total_loss: the average loss per example
        """
        np.random.seed(0)
        torch.manual_seed(0)
        # torch.use_deterministic_algorithms(True)
        if self.start_time is None:
            self.start_time = time.time()

        # The following adjustment in the loss computation
        # was made, becase torch.nn.CrossEntropyLoss returns
        # the avg. loss for each batch
        # epoch_size = 0
        batch_count = 0
        batch_size = 0
        total_loss = 0
        for batch_idx, batch_data in enumerate(data_loader):
            encoder_x, decoder_x, y = (
                batch_data["input_ids"],
                batch_data["decoder_input_ids"],
                batch_data["labels"],
            )
            if batch_idx == 0:
                batch_size = encoder_x.size(0)
            # epoch_size += encoder_x.size(0)
            batch_count += encoder_x.size(0) / batch_size
            loss = self.run_one_batch(
                encoder_x, decoder_x, y, train=train, pad_token_id=pad_token_id
            )
            total_loss += loss

        avg_loss = total_loss / batch_count

        if verbose:
            epoch = self.epoch + 1
            duration = (time.time() - self.start_time) / 60

            if train:
                log = [f"Epoch: {epoch:6d}"]
            else:
                log = ["Eval:" + " " * 8]

            log.extend(
                [
                    f"Loss: {avg_loss:6.3f}",
                    f"in {duration:5.1f} min",
                ]
            )
            print("  ".join(log))

        return avg_loss

    def train(
        self, data_loader, n_epochs, train=True, report_every=None, pad_token_id=0
    ):
        """
        Run the model for `n_epochs` epochs on the data in `data_loader`
        You shouldn't need to edit this function.

        Args
            data_loader: data loader for our data
            n_epochs: how many epochs to run
            train: if True, train the model; otherwise, just evaluate it
            report_every: how often to print out stats

        Returns
            losses: average loss per epoch
        """
        self.start_time = time.time()

        if report_every is None:
            report_every = max(1, n_epochs // 10)

   
        losses = []
        for i in range(n_epochs):
            verbose = ((i + 1) % report_every) == 0
            loss = self.run_one_epoch(
                data_loader["train"],
                train=train,
                verbose=verbose,
                pad_token_id=pad_token_id,
            )
            losses.append(loss)
            if train:
                self.epoch += 1
                self.eval({"train": data_loader["validation"]})

        return losses

    def eval(self, data_loader):
        """
        Helper function to run through the data loader once and just
            compute the loss
        You shouldn't need to edit this function.
        """
        return self.train(data_loader, 1, train=False, report_every=1)

    def save_trainer(self, filename):
        """
        Use `src.utils.save` to save this Trainer to file.
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html

        Args
            filename: the file to which to save the trainer
        """
        save(
            filename,
            epoch=self.epoch,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )

    def load_trainer(self, filename):
        """
        Use `src.utils.load` to load this trainer from file.
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html

        Note: in addition to simply loading the saved model, you must
            use the information from that checkpoint to update the model's
            state.

        Args
            filename: the file from which to load the model
        """
        checkpoint = load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]

        # this ensures that the trainer's/model's parameters are in training mode
        self.model.train()



from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

batch_size = 100

# print(tokenizer.vocab_size)
train_data_loader = DataLoader(wmt14_train_subset, batch_size, shuffle=True)
test_data_loader = DataLoader(wmt14_test_subset, batch_size, shuffle=True)
valdiation_data_loader = DataLoader(
    wmt14_validation_subset, batch_size, shuffle=True
)

model = Transformer(
    batch_size=100,
    seq_len=32,
    d_model=32,
    number_of_heads=4,
    # +1 for bos token, it is documented that
    # tokenizer.vocab_size does not include the
    # added tokens
    input_vocab_size=tokenizer.vocab_size + 1,
    output_vocab_size=tokenizer.vocab_size + 1,
    dropout_p=0.1,
).to(device)

d_model = 64


def lr_lambda(step_num):
    if step_num == 0:
        step_num = 1
    return d_model ** (-0.5) * step_num ** (-0.5)


trainer_args = {
    "lr": 1.0,
    "optimizer": Adam,
}

scheulder = {"scheduler": lr_scheduler.LambdaLR, "lr_lambda": lr_lambda}
trainer = Trainer(
    model=model, loss_func=CrossEntropyLoss(), scheduler=scheulder, **trainer_args
)

# Create a dictionary of data loaders
data_loader = {
    "test": test_data_loader,
    "train": train_data_loader,
    "validation": valdiation_data_loader,
}

trainer.train(data_loader=data_loader, n_epochs=500, pad_token_id=tokenizer.pad_token_id)
# trainer.run_one_epoch(
#     data_loader=data_loader["train"], verbose=True, pad_token_id=tokenizer.pad_token_id
# )
# first_data = next(iter(data_loader))
# print(first_data["attention_mask"].size())