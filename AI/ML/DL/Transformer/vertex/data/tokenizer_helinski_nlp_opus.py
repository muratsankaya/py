"""
This module provides functionality to preprocess and tokenize examples from 
the WMT14 dataset for training a transformer model.
"""

from data.wmt_english_german import wmt14_train, wmt14_test, wmt14_validation
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

    decoder_input_ids = tokenized_targets["input_ids"]

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
        [bos_id] + [seq_id for seq_id in seq if seq_id != eos_id]
        for seq in decoder_input_ids
    ]

    # When indicies are set to -100 they are ignored in loss computation of cross entropy
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    labels = [
        [(seq_id if seq_id != pad_id else -100) for seq_id in seq]
        for seq in decoder_input_ids
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
wmt14_train_subset = preprocess_dataset(wmt14_train, num_of_examples=128)
print(f"train dataset size: {len(wmt14_train_subset)}")

print("Printing preprocessed test dataset:")
wmt14_test_subset = preprocess_dataset(wmt14_test, num_of_examples=64)
print(f"test dataset size: {len(wmt14_test_subset)}")

print("Printing preprocessed validation dataset:")
wmt14_validation_subset = preprocess_dataset(wmt14_validation, num_of_examples=64)
print(f"validation dataset size: {len(wmt14_validation_subset)}")
