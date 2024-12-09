import argparse
import numpy as np
import os
import time
import torch

from src.mlp import MLP
from src.trainer import Trainer
from src.data import AddDataset
from src.experiments import params_add_dataset


def add_dataset_experiment(num_examples=1000, batch_size=100):
    """
    This is a copy of the `test_add_dataset` case in tests/test_model.py,
        set up to allow you easily tweak the batch size and dataset size.
    Run this script from the root directory of your repository with:
        `python -m free_response.batch_sizes <num_examples> <batch_size>`
    """
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dataset = AddDataset(num_examples=num_examples)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    model_args, trainer_args = params_add_dataset()

    model = MLP(**model_args)
    model.initialize()

    trainer = Trainer(
        model=model,
        loss_func=torch.nn.MSELoss(),
        **trainer_args)

    start_time = time.time()
    trainer.train(data_loader, 100, report_every=100)
    end = time.time() - start_time
    print(f"Training took {end:.1f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_examples", type=int)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    add_dataset_experiment(args.num_examples, args.batch_size)
