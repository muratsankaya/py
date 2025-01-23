import torch


def make_onehot(arr, vocab_size):
    # Convert to one-hot representation
    # https://en.wikipedia.org/wiki/One-hot

    n, max_length = arr.shape
    onehot_data = np.zeros([n, max_length, vocab_size])
    for v in range(vocab_size):
        onehot_row = np.zeros([vocab_size])
        onehot_row[v] = 1
        onehot_data[arr == v] = onehot_row

    return onehot_data


class CountingDataset(torch.utils.data.Dataset):
    def __init__(self, n, max_length=8, vocab_size=8):

        assert vocab_size > 2
        self.n = n
        self.vocab_size = vocab_size
        seq_lengths = np.random.randint(max_length // 2, max_length, n)
        data = np.random.randint(0, vocab_size, [n, max_length])

        # Replace elements past the sequence length with -1
        for i in range(n):
            data[i, slice(seq_lengths[i] + 1, None)] = -1

        onehot_data = make_onehot(data, vocab_size)

        # Label is whether ones outnumber twos in the sequence
        num_ones = (data == 1).sum(axis=1, keepdims=True)
        num_twos = (data == 2).sum(axis=1, keepdims=True)
        label = (num_ones > num_twos).astype(int)

        self.data = torch.tensor(onehot_data).float()
        self.label = torch.tensor(label).long()

    def __len__(self):
        return self.n

    def __getitem__(self, item_index):
        """
        Allow us to select items with `dataset[0]`
        Returns (x, y)
            x: the data tensor
            y: the label tensor
        """
        return self.data[item_index], self.label[item_index]


d = CountingDataset(3, max_length=8, vocab_size=3)
d[0]
