import os
import torch


def save(filename, **kwargs):
    """
    Save a pytorch object to file
    See: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    You shouldn't need to edit this function.

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
