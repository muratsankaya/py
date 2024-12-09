import numpy as np
import os
import torch

from src.mlp import MLP
from src.trainer import Trainer
from src.data import AddDataset, MultiplyDataset
from src.experiments import params_add_dataset, params_multiply_dataset


def main():
    """
    This is a demo function provided just to highlight how you might
    train and save your models to pass the `test_saved_add_dataset`
    and `test_saved_multiply_dataset` cases.

    You may want to modify the number of examples and number of training
    epochs used in `trainer.train(...)`.
    """

    model_type = input(
        "Which model would you like to train and then save?(add/multiply) "
    )

    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    outfn = (
        "models/test_saved_add_dataset.pt"
        if model_type == "add"
        else "models/test_saved_multiply_dataset.pt"
    )
    if os.path.exists(outfn):
        choice = input(f"Delete {outfn}? y/n ")
        if choice == "y":
            os.remove(outfn)

    dataset = (
        AddDataset(num_examples=1000)
        if model_type == "add"
        else MultiplyDataset(num_examples=1000)
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

    model_args, trainer_args = (
        params_add_dataset() if model_type == "add" else params_multiply_dataset()
    )
    model = MLP(**model_args)
    model.initialize()

    trainer = Trainer(model=model, loss_func=torch.nn.MSELoss(), **trainer_args)

    _ = trainer.train(data_loader, 200 if model_type == "add" else 200)
    losses = trainer.eval(data_loader)

    model.save_model(outfn)


if __name__ == "__main__":
    main()
