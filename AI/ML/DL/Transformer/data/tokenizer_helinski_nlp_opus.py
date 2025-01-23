"""
This module provides functionality to preprocess and tokenize examples from 
the WMT14 dataset for training a transformer model.

The WMT14 dataset is a collection of parallel sentences in German and English. 
The dataset is structured as follows:

DatasetDict({
    train: Dataset({
        features: ['translation'],
        num_rows: 4508785
    })
    validation: Dataset({
        features: ['translation'],
        num_rows: 3000
    })
    test: Dataset({
        features: ['translation'],
        num_rows: 3003
    })
})

Each example in the dataset is a dictionary with a 'translation' key, which 
contains a dictionary with 'de' (German) and 'en' (English) keys:

{'translation': {'de': 'Wiederaufnahme der Sitzungsperiode', 'en': 'Resumption of the session'}}
{'translation': {'de': 'Ich erkläre die am Freitag, dem 17. Dezember unterbrochene Sitzungsperiode 
des Europäischen Parlaments für wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum Jahreswechsel 
und hoffe, daß Sie schöne Ferien hatten.', 'en': 'I declare resumed the session of the European Parliament 
adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope 
that you enjoyed a pleasant festive period.'}}

This module converts these examples into tokenized inputs and decoder inputs suitable for 
training a transformer model. The preprocessing steps include:

1. Loading a pre-trained English-German tokenizer.
2. Determining the maximum sequence length for tokenized sequences.
3. Defining a preprocessing function to tokenize the source (German) 
and target (English) sentences.
4. Applying the preprocessing function to a subset of the training set.
5. Converting the tokenized examples to PyTorch tensors for consistency.

Example output after preprocessing:
Converts examples from the WMT14 dataset to tokenized inputs and decoder inputs

# Example
Example 0
Input IDs: tensor([423, 42, 11967, 2985, 2, 29, 9, 372, 4209, 8319])
Decoder Input IDs: tensor([4677, 2, 52, 41, 73, 72, 1770, 2, 4, 48186])
Decoded decoder input: Although, as you will have seen, the dreaded 
'millennium bug' failed to materialise...

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
