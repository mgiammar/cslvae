"""
Model for predicting properties from latent space. Implements MLP both with and without
time dependence (for working alongside a diffusion model for property-guided sampling).
TODO: Docstring
"""


import os
import ast
from typing import List, Union, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from latent_diffusion.time_embedding import TimeEmbedding
from latent_diffusion.noise_scheduler import DiscreteNoiseScheduler


class PropertyModel(nn.Module):
    """Base class for property models. Implements a simple MLP with time dependence
    (for working alongside a diffusion model for property-guided sampling) or just as a
    simple model for predicting p(y|z).

    """

    @classmethod
    def parse_config(cls, config: dict):
        """Parses the configuration dictionary for the property model.

        Arguments:
            (dict) config: The configuration dictionary for the property model.
        """
        REQUIRED_KEYS = {
            "input_dim",
            "output_dim",
            "num_hidden_layers",
            "hidden_layer_shapes",
            # "activation_cls",  # Not required since ReLU is default
            # "activation_kwargs",
        }

        # Check for required keys
        missing_keys = REQUIRED_KEYS - set(config.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys in config dictionary: {missing_keys}")

        input_dim = int(config["input_dim"])
        output_dim = int(config["output_dim"])
        num_hidden_layers = int(config["num_hidden_layers"])
        hidden_layer_shapes = config["hidden_layer_shapes"]
        if isinstance(hidden_layer_shapes, str):
            hidden_layer_shapes = ast.literal_eval(hidden_layer_shapes)

        # Activation class and keyword arguments
        activation_cls = config.get("activation_cls", "ReLU")
        activation_cls = getattr(torch.nn, activation_cls)
        activation_kwargs = config.get("activation_kwargs", {})
        if isinstance(activation_kwargs, str):
            activation_kwargs = ast.literal_eval(activation_kwargs)

        return PropertyModel(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_shapes=hidden_layer_shapes,
            activation_cls=activation_cls,
            activation_kwargs=activation_kwargs,
        )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        hidden_layer_shapes: List[int],
        activation_cls: Union[str, nn.Module],
        activation_kwargs: dict,
    ):
        super().__init__()

        # Check for valid inputs
        if num_hidden_layers != len(hidden_layer_shapes):
            raise ValueError(
                f"num_hidden_layers must equal the length of hidden_layer_shapes. Got "
                f"{num_hidden_layers} and {len(hidden_layer_shapes)}, respectively."
            )

        if len(hidden_layer_shapes) == 0:
            raise ValueError("hidden_dims must have at least one element.")

        # Get the activation function and class based on passed arguments
        activation_kwargs = activation_kwargs if activation_kwargs is not None else {}
        try:
            activation_cls = (
                getattr(torch.nn, activation_cls)
                if isinstance(activation_cls, str)
                else activation_cls
            )
        except AttributeError:
            raise ValueError(f"Unrecognized torch activation: {activation_cls}")

        # Add all layers to a sequential module
        modules = []
        modules.append(nn.Linear(input_dim, hidden_layer_shapes[0]))
        modules.append(activation_cls(**activation_kwargs))

        for i in range(len(hidden_layer_shapes) - 1):
            modules.append(
                nn.Linear(hidden_layer_shapes[i], hidden_layer_shapes[i + 1])
            )
            modules.append(activation_cls(**activation_kwargs))

        modules.append(nn.Linear(hidden_layer_shapes[-1], output_dim))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)

    def _train_single_epoch(
        self, dataloader, epoch, optimizer, criterion, device
    ) -> float:
        """Trains a single epoch of the property model returning the average loss value
        over the epoch.
        TODO: Complete docstring
        """
        tmp_loss = []

        # NOTE: Already expected that the target property has shape of
        # (batch_size, n) where n are the number of properties. The property vector
        # is still 2D even if n = 1.
        for batch_idx, (feature, target) in enumerate(dataloader):
            feature = feature.to(device)
            target = target.to(device)

            # Predict the target property value(s)
            prediction = self.forward(feature)

            # Compute loss and take gradient step
            optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            # Append mini-batch loss to tracked losses
            tmp_loss.append(loss.item())

        return np.mean(tmp_loss)

    def _compute_test_loss(self, dataloader, criterion, device) -> float:
        """Computes the test loss of the property model."""
        tmp_loss = []
        with torch.no_grad():
            for batch_idx, (feature, target) in enumerate(dataloader):
                feature = feature.to(device)
                target = target.to(device)

                # Predict the target property value(s)
                prediction = self.forward(feature)

                loss = criterion(prediction, target)
                tmp_loss.append(loss.item())

        return np.mean(tmp_loss)

    def _train_logging_function(
        self, logging_iterations, epoch, num_epochs, metrics_dict, writer
    ) -> None:
        """Helper function for logging training metrics to the console and tensorboard
        during training when epoch is a multiple of logging_iterations, or if it is the
        final epoch. If writer is None, no tensorboard logs are generated.

        Arguments:
            (int) logging_iterations: How often to print out training information
            (int) epoch: The current epoch
            (int) num_epochs: The total number of epochs to train the model
            (dict) metrics_dict: A dictionary of metrics to log
            (SummaryWriter) writer: An optional SummaryWriter object to use for logging
                to tensorboard. If None, no tensorboard logs are generated.
        """
        # Skip logging if not at a logging iteration or final epoch
        if epoch % logging_iterations != 0 and epoch != num_epochs - 1:
            return

        train_loss = metrics_dict["train_loss"]
        test_loss = metrics_dict["test_loss"]
        print(f"Finished epoch {epoch + 1} / {num_epochs}\t", end="")
        print(f"Train Loss: {train_loss:.6f}\t", end="")
        print(f"Test Loss: {test_loss:.6f}")

        # Log metrics to tensorboard, if requested
        if writer is not None:
            for metric_key, metric_value in metrics_dict.items():
                writer.add_scalar(f"{metric_key}", metric_value, epoch)

    def _train_checkpoint_function(
        self, checkpoint_iterations, epoch, optimizer, metrics_dict, outdir
    ) -> None:
        """Helper function for logging model checkpoints during training at epochs
        which are multiples of checkpoint_iterations. If outdir is None, no checkpoints
        are saved.

        Arguments:
            (int) checkpoint_iterations: How often to save model checkpoints
            (int) epoch: The current epoch
            (torch.optim.Optimizer) optimizer: The optimizer object
            (dict) metrics_dict: A dictionary of metrics to log
            (str) outdir: The directory to save model checkpoints and tensorboard logs
                to. If None, no checkpoints or logs are saved.
        """
        if outdir is None:
            return

        # Skip checkpointing if not at a checkpoint iteration
        if epoch % checkpoint_iterations != 0:
            return

        checkpoint_path = os.path.join(outdir, f"model_checkpoint_{epoch}.pth")
        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": metrics_dict["train_loss"],
            "test_loss": metrics_dict["test_loss"],
        }
        torch.save(state_dict, checkpoint_path)

    def fit(
        self,
        dataset: Dataset,
        batch_size: int,
        num_epochs: int,
        logging_iterations: int,
        checkpoint_iterations: int,
        optimizer_cls: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        criterion_cls: torch.nn.modules.loss._Loss,
        outdir: str,
        shuffle_train_dl: bool = True,
        shuffle_test_dl: bool = False,
        train_test_split: Tuple[int, int] = (0.8, 0.2),
        log_to_tensorboard: bool = True,
    ):
        self.train()
        device = next(self.parameters()).device
        writer = SummaryWriter(log_dir=outdir) if log_to_tensorboard else None

        # Optimizer from config attributes
        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        criterion = criterion_cls()

        # Split the dataset into train and test set and instantiate dataloaders
        # NOTE: Could fix a generator here for reproductivity sake
        train_dataset, test_dataset = random_split(dataset, train_test_split)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train_dl,
            num_workers=0,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_test_dl,
            num_workers=0,
            pin_memory=True,
        )

        # Iterate over epochs and train the model
        for epoch in range(num_epochs):
            train_loss = self._train_single_epoch(
                train_dataloader, epoch, optimizer, criterion, device
            )
            test_loss = self._compute_test_loss(test_dataloader, criterion, device)

            # Construct metrics dictionary
            metrics_dict = {"train_loss": train_loss, "test_loss": test_loss}

            # Logging and checkpointing functions
            self._train_logging_function(
                logging_iterations, epoch, num_epochs, metrics_dict, writer
            )
            self._train_checkpoint_function(
                checkpoint_iterations, epoch, optimizer, metrics_dict, outdir
            )

        # Close tensorboard writer
        if writer is not None:
            writer.close()

        # Save final model state
        final_model_path = os.path.join(outdir, "model_checkpoint_final.pth")
        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        torch.save(state_dict, final_model_path)


class TimeDependentPropertyModel(PropertyModel):
    """Property model with time dependence (for working alongside a diffusion model for
    property-guided sampling).

    TODO: Complete docstring

    NOTE: Does testing with adding noise to the input features make sense? Loss at
    different time samples?

    """

    @classmethod
    def parse_config(cls, config: dict):
        """Parses the configuration dictionary for the property model.

        Arguments:
            (dict) config: The configuration dictionary for the property model.
        """
        REQUIRED_KEYS = {
            "input_dim",
            "output_dim",
            "time_embedding_dim",
            "num_hidden_layers",
            "hidden_layer_shapes",
            # "activation_cls",  # Not required since ReLU is default
            # "activation_kwargs",
            "time_embedding",
            "noise_scheduler",
        }

        # Check for required keys
        missing_keys = REQUIRED_KEYS - set(config.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys in config dictionary: {missing_keys}")

        input_dim = int(config["input_dim"])
        time_embedding_dim = int(config["time_embedding_dim"])
        output_dim = int(config["output_dim"])
        num_hidden_layers = int(config["num_hidden_layers"])
        hidden_layer_shapes = config["hidden_layer_shapes"]
        if isinstance(hidden_layer_shapes, str):
            hidden_layer_shapes = ast.literal_eval(hidden_layer_shapes)

        # Activation class and keyword arguments
        activation_cls = config.get("activation_cls", "ReLU")
        activation_cls = getattr(torch.nn, activation_cls)
        activation_kwargs = config.get("activation_kwargs", {})
        if isinstance(activation_kwargs, str):
            activation_kwargs = ast.literal_eval(activation_kwargs)

        # Parse the time embedding and noise scheduler
        time_embedding = TimeEmbedding.parse_config(config["time_embedding"])
        noise_scheduler = DiscreteNoiseScheduler.parse_config(config["noise_scheduler"])

        return TimeDependentPropertyModel(
            input_dim=input_dim,
            time_embedding_dim=time_embedding_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_shapes=hidden_layer_shapes,
            activation_cls=activation_cls,
            activation_kwargs=activation_kwargs,
            time_embedding=time_embedding,
            noise_scheduler=noise_scheduler,
        )

    def __init__(
        self,
        input_dim: int,
        time_embedding_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        hidden_layer_shapes: List[int],
        activation_cls: Union[str, nn.Module],
        activation_kwargs: dict,
        time_embedding: TimeEmbedding,
        noise_scheduler: TimeEmbedding,
    ):
        if time_embedding_dim != time_embedding.dim:
            raise ValueError(
                "The `time_embedding_dim` argument must match the `time_embedding` "
                f"object's `dim` attribute. Got {time_embedding_dim} and "
                f"{time_embedding.dim}, respectively."
            )

        # Instantiate the base property model class
        super().__init__(
            input_dim=input_dim + time_embedding_dim,  # Add time embedding dimension
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_shapes=hidden_layer_shapes,
            activation_cls=activation_cls,
            activation_kwargs=activation_kwargs,
        )

        # Hold the time embedding and noise scheduler classes
        self.time_embedding = time_embedding
        self.noise_scheduler = noise_scheduler

    def _train_single_epoch(
        self, dataloader, epoch, optimizer, criterion, device
    ) -> float:
        """Trains a single epoch of the property model returning the average loss value
        over the epoch.
        TODO: Complete docstring
        """
        tmp_loss = []

        # NOTE: Already expected that the target property has shape of
        # (batch_size, n) where n are the number of properties. The property vector
        # is still 2D even if n = 1.
        for batch_idx, (feature, target) in enumerate(dataloader):
            # Sample time points uniformly from the noise scheduler
            t = torch.randint(
                low=0,
                high=self.noise_scheduler.T,
                size=(feature.shape[0],),
                device=device,
            )
            t_emb = self.time_embedding.get_embedding(t).to(device)

            # Add noise to the feature vector
            noised_feature, eps = self.noise_scheduler.add_noise_to_sample(feature, t)
            model_input = torch.cat([noised_feature, t_emb], dim=1)
            target = target.to(device)

            # Predict the target property value(s) from the noised feature vector
            prediction = self.forward(model_input)

            # Compute the loss and take a gradient step
            optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            # Append mini-batch loss to tracked losses
            tmp_loss.append(loss.item())

        return np.mean(tmp_loss)

    def _compute_test_loss(self, dataloader, criterion, device) -> float:
        """Computes the test loss of the property model.
        TODO: Docstring
        """
        tmp_loss = []

        with torch.no_grad():
            for batch_idx, (feature, target) in enumerate(dataloader):
                # Sample time points uniformly from the noise scheduler
                t = torch.randint(
                    low=0,
                    high=self.noise_scheduler.T,
                    size=(feature.shape[0],),
                    device=device,
                )
                t_emb = self.time_embedding.get_embedding(t).to(device)

                # Add noise to the feature vector
                noised_feature, eps = self.noise_scheduler.add_noise_to_sample(
                    feature, t
                )
                model_input = torch.cat([noised_feature, t_emb], dim=1)
                target = target.to(device)

                # Predict the target property value(s) from the noised feature vector
                prediction = self.forward(model_input)

                loss = criterion(prediction, target)
                tmp_loss.append(loss.item())

        return np.mean(tmp_loss)

    def fit(
        self,
        dataset: Dataset,
        batch_size: int,
        num_epochs: int,
        logging_iterations: int,
        checkpoint_iterations: int,
        optimizer_cls: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        criterion_cls: torch.nn.modules.loss._Loss,
        outdir: str,
        shuffle_train_dl: bool = True,
        shuffle_test_dl: bool = False,
        train_test_split: Tuple[int, int] = (0.8, 0.2),
        log_to_tensorboard: bool = True,
    ) -> None:
        """TODO: Docstring"""

        # TODO: Reduce code duplication between this and the base class
        self.train()
        device = next(self.parameters()).device
        writer = SummaryWriter(log_dir=outdir) if log_to_tensorboard else None

        # Optimizer from config attributes
        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        criterion = criterion_cls()

        # Split the dataset into train and test set and instantiate dataloaders
        # NOTE: Could fix a generator here for reproductivity sake
        train_dataset, test_dataset = random_split(dataset, train_test_split)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train_dl,
            num_workers=0,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_test_dl,
            num_workers=0,
            pin_memory=True,
        )

        # Iterate over epochs and train the model
        for epoch in range(num_epochs):
            train_loss = self._train_single_epoch(
                train_dataloader, epoch, optimizer, criterion, device
            )
            test_loss = self._compute_test_loss(test_dataloader, criterion, device)

            # Construct metrics dictionary
            metrics_dict = {"train_loss": train_loss, "test_loss": test_loss}

            # Logging and checkpointing functions
            self._train_logging_function(
                logging_iterations, epoch, num_epochs, metrics_dict, writer
            )
            self._train_checkpoint_function(
                checkpoint_iterations, epoch, optimizer, metrics_dict, outdir
            )

        # Close tensorboard writer
        if writer is not None:
            writer.close()

        # Save final model state
        final_model_path = os.path.join(outdir, "model_checkpoint_final.pth")
        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        torch.save(state_dict, final_model_path)
