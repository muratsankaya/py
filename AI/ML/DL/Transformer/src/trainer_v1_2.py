import numpy as np
import torch
from torcheval.metrics.functional.text import perplexity
from src.utils import save, load
import time

device = "cpu"
dtype = torch.float32


class Trainer:
    def __init__(self, optimizer, model, loss_func, scheduler=None, **kwargs):
        """
        Initialize the optimizer for the model, using any necessary kwargs
        Save the model and loss function for later calculation
        You shouldn't need to edit this function.
        """

        self.optimizer: torch.optim.Optimizer = optimizer(model.parameters(), **kwargs)
        self.scheduler = scheduler["scheduler"](
            self.optimizer, lr_lambda=scheduler["lr_lambda"]
        )
        self.model: torch.nn.Module = model
        self.loss_func = loss_func

        self.epoch = 0
        self.start_time = None

    def run_one_batch(self, encder_x, decoder_x, y, train=True, pad_token_id=0):
        """
        Run self.model on one batch of data, using `self.loss_func` to
            compute the model's loss.

        If train=True (the default), you should use `self.optimizer`
            to update the parameters of `self.model`.

        You should also call `self.optimizer.zero_grad()`; see
            https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            for a guide as to when to do that.

        Args
            enocder_x: the batch's enocder input
            decoder_x: the batch's decoder input
            y: the batch's target

        Returns
            loss: the model's loss on this batch
        """

        if train:
            self.optimizer.zero_grad()

        outputs = self.model(
            encder_x.to(device), decoder_x.to(device), pad_token_id=pad_token_id
        )
        loss = self.loss_func(outputs.transpose(1, 2), y.to(device))
        perplexity_score = perplexity(outputs, y.to(device), ignore_index=-100)

        if train:
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return loss.detach().cpu().numpy(), perplexity_score.detach().cpu().numpy()

    def run_one_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        train=True,
        verbose=False,
        pad_token_id=0,
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
        # torch.use_deterministic_algorithms(True)
        if self.start_time is None:
            self.start_time = time.time()

        # The following adjustment in the loss computation
        # was made, becase torch.nn.CrossEntropyLoss returns
        # the avg. loss for each batch
        # epoch_size = 0
        batch_count = 0
        batch_size = 0
        total_loss = 0
        total_perplexity_score = 0
        for batch_idx, batch_data in enumerate(data_loader):
            encoder_x, decoder_x, y = (
                batch_data["input_ids"],
                batch_data["decoder_input_ids"],
                batch_data["labels"],
            )
            if batch_idx == 0:
                batch_size = encoder_x.size(0)
            # epoch_size += encoder_x.size(0)
            batch_count += encoder_x.size(0) / batch_size
            loss, perplexity_score = self.run_one_batch(
                encoder_x, decoder_x, y, train=train, pad_token_id=pad_token_id
            )
            total_loss += loss
            total_perplexity_score += perplexity_score

        avg_loss = total_loss / batch_count
        avg_perplexity_score = total_perplexity_score / batch_count

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
                    f"Perpelixty Score: {avg_perplexity_score:6.3f}",
                    f"in {duration:5.1f} min",
                ]
            )
            print("  ".join(log))

        return avg_loss

    def train(
        self, data_loader, n_epochs, train=True, report_every=None, pad_token_id=0
    ):
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
        if train:
            self.start_time = time.time()

        if report_every is None:
            report_every = max(1, n_epochs // 10)

        losses = []
        for i in range(n_epochs):
            verbose = ((i + 1) % report_every) == 0
            loss = self.run_one_epoch(
                data_loader["train"],
                train=train,
                verbose=verbose,
                pad_token_id=pad_token_id,
            )
            losses.append(loss)
            if train:
                self.epoch += 1
                if verbose:
                    self.eval({"train": data_loader["validation"]})

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
