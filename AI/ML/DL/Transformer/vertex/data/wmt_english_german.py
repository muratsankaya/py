"""
Loads the wmt14 dutch-to-english (de-en) variable from huggging 
face or cache (if its already downloads before)
and stores it in a variable called wmt14
"""

from datasets import load_dataset


wmt14_train = load_dataset(
    "parquet",
    data_files={
        "train": [
            "data/gcs/train/train-00000-of-00003.parquet",
            "data/gcs/train/train-00001-of-00003.parquet",
            "data/gcs/train/train-00002-of-00003.parquet",
        ]
    },
    split="train",
)
wmt14_test = load_dataset(
    "parquet",
    data_files={"test": "data/gcs/test/test-00000-of-00001.parquet"},
    split="test",
)
wmt14_validation = load_dataset(
    "parquet",
    data_files={"validation": "data/gcs/validation/validation-00000-of-00001.parquet"},
    split="validation",
)


# Display the first 5 rows of the 'train' split

print(f"\nThe train dataset has {wmt14_train.num_rows} examples")
print("printing 2 examples from the train dataset:")
for i in range(2):
    print(wmt14_train[i])

print(f"\nThe test dataset has {wmt14_test.num_rows} examples")
print("printing 2 examples from the test dataset:")
for i in range(2):
    print(wmt14_test[i])

print(f"\nThe validation dataset has {wmt14_validation.num_rows} examples")
print("printing 2 examples from the validation dataset")
for i in range(2):
    print(wmt14_validation[i])
