"""
Loads the wmt12 german-to-englis (de-en) variable from huggging 
face or cache (if its already downloads before)
and stores it in a variable called wmt14

"""

from datasets import load_dataset

wmt14_train = load_dataset("wmt14", "de-en", split="train")
wmt14_test = load_dataset("wmt14", "de-en", split="test")
wmt14_validation = load_dataset("wmt14", "de-en", split="validation")

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

# # Convert the 'train' split to a DataFrame
# df_train = pd.DataFrame(wmt14["train"])

# # Display the first 5 rows
# print(df_train.head())

# # Calculate sentence lengths
# en_lengths = [len(ex["translation"]["en"].split()) for ex in wmt14["train"]]
# de_lengths = [len(ex["translation"]["de"].split()) for ex in wmt14["train"]]

# # Plot histograms
# plt.hist(en_lengths, bins=50, alpha=0.5, label="English")
# plt.hist(de_lengths, bins=50, alpha=0.5, label="German")
# plt.xlabel("Sentence Length")
# plt.ylabel("Frequency")
# plt.legend(loc="upper right")
# plt.title("Sentence Length Distribution")
# plt.show()
