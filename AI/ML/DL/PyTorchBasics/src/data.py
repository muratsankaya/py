import torch
import numpy as np


class AddDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples):
        """
        Create a dataset of the form x_1 + x_2 = y

        Save the dataset to class variables.
        You should use torch tensors of dtype float32.
        """
        self.num_examples = num_examples
        data = np.random.randint(-1000, 1000, size=[num_examples, 2])
        label = data.sum(axis=1, keepdims=True)

        # TODO Convert to torch tensors and save these as class variables
        # so we can load them with self.__getitem__
        self.X = torch.from_numpy(data).to(torch.float32)
        self.y = torch.from_numpy(label).to(torch.float32)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item_index):
        """
        Allow us to select items with `dataset[0]`
        Use the class variables you created in __init__.

        Returns (x, y)
            x: the data tensor
            y: the label tensor
        """
        return (self.X[item_index], self.y[item_index])


class MultiplyDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples):
        """
        Create a dataset of the form x_1 * x_2 = y

        Save the dataset to class variables.
        You should use torch tensors of dtype float32.
        """
        self.num_examples = num_examples
        data = np.random.randint(1, 1000, size=[num_examples, 2])
        label = data.prod(axis=1, keepdims=True)

        # just to test label tensor should have the shape (num_examples, 1)(CORRECT)
        # print(label.shape)

        # without keepdims it will shrink axis 1 in this case column
        # and the shape would be num_examples (CORRECT)
        # label1 = data.prod(axis=1)
        # print(label1.shape)

        # TODO Convert to torch tensors and save these as class variables
        #  so we can load them with self.__getitem__
        self.X = torch.from_numpy(data).to(torch.float32)
        self.y = torch.from_numpy(label).to(torch.float32)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item_index):
        """
        Allow us to select items with `dataset[0]`
        Returns (x, y)
            x: the data tensor
            y: the label tensor
        """
        return (self.X[item_index], self.y[item_index])
