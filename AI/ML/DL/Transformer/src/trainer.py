import time

import numpy as np
import torch

from src.utils import save, load


class Trainer:
    def __init__(self, optimizer, model, loss_func, **kwargs):
        """
        Initialize the optimizer for the model, using any necessary kwargs
        Save the model and loss function for later calculation
        You shouldn't need to edit this function.
        """

        self.optimizer: torch.optim.Optimizer = optimizer(model.parameters(), **kwargs)
        self.model: torch.nn.Module = model
        self.loss_func = loss_func

        self.epoch = 0
        self.start_time = None

    def run_one_batch(self, x, y, train=True):
        """
        Run self.model on one batch of data, using `self.loss_func` to
            compute the model's loss.

        If train=True (the default), you should use `self.optimizer`
            to update the parameters of `self.model`.

        You should also call `self.optimizer.zero_grad()`; see
            https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            for a guide as to when to do that.

        Args
            x: the batch's input
            y: the batch's target

        Returns
            loss: the model's loss on this batch
        """

        if train:
            self.optimizer.zero_grad()

        outputs = self.model(x)
        loss = self.loss_func(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss.detach().numpy()

    def run_one_epoch(
        self, data_loader: torch.utils.data.DataLoader, train=True, verbose=False
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
        torch.use_deterministic_algorithms(True)
        if self.start_time is None:
            self.start_time = time.time()

        epoch_size = 0
        total_loss = 0
        for batch_idx, batch_data in enumerate(data_loader):
            x, y = batch_data
            epoch_size += x.size(0)
            loss = self.run_one_batch(x, y, train=train)
            total_loss += loss

        avg_loss = total_loss / epoch_size

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

    def train(self, data_loader, n_epochs, train=True, report_every=None):
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
            loss = self.run_one_epoch(data_loader, train=train, verbose=verbose)
            losses.append(loss)
            if train:
                self.epoch += 1

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
