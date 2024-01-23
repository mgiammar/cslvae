from typing import List
import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from latent_diffusion.dataset import LatentAndPropertyDataset
from latent_diffusion.noise_scheduler import DiscreteNoiseScheduler
from latent_diffusion.time_embedding import TimeEmbedding
from latent_diffusion.utils import subtract_scaled_noise


# TODO: Better logic for keeping all tensors the same datatype as the model expects
# TODO: Better logic for sending tensors to/initializing tensors on the proper device


class PropertyGuidedDDPM(nn.Module):
    """Latent Diffusion Denoising Probabilistic Model for performing diffusion in latent
    CSLVAE space. This class can can accept in a time-dependent property prediction
    model during the denoising process (TODO: Add checks for paired model input-output
    and noise schedulers) to guide sampled latent space towards points with properties
    of interest.

    Currently implemented as a MLP with time embedding.  TODO: Allow other archs.

    TODO: Docstring

    """

    def __init__(
        self,
        latent_dim: int,
        time_embedding_dim: int,
        num_layers: int,
        hidden_dims: List[int],
        time_embedding: TimeEmbedding,
        noise_scheduler: DiscreteNoiseScheduler,
        activation: nn.Module = nn.ReLU(),  # NOTE: Does this cause a problem with duplicated instances?
    ):
        super().__init__()

        # Check for valid inputs
        if num_layers != len(hidden_dims):
            raise ValueError(
                f"num_layers must equal the length of hidden_dims. Got {num_layers} "
                f"and {len(hidden_dims)}, respectively."
            )

        # Add all layers to a sequential module
        modules = []
        modules.append(nn.Linear(latent_dim + time_embedding_dim, hidden_dims[0]))
        modules.append(activation)

        for i in range(num_layers - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(activation)

        modules.append(nn.Linear(hidden_dims[-1], latent_dim))

        self.sequential = nn.Sequential(*modules)

        # Store time embedding and noise scheduler attributes
        self.time_embedding = time_embedding
        self.noise_scheduler = noise_scheduler

    def forward(self, x):
        return self.sequential(x)

    def _single_denoise_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        alphas: torch.Tensor,
        alphas_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """Performs a single denoising step on a sample x from time t to t-1. Note that
        time must be already cast to a tensor; integer t is not accepted.

        TODO: Incorporate property guidance into the denoising process

        Arguments:
            (torch.Tensor) x: The sample tensor to denoise
            (torch.Tensor) t: The time-step in the noising process to denoise from.
            (torch.Tensor) alphas: The alphas coefficients from the noise scheduler
            (torch.Tensor) alphas_tilde: The alphas_tilde coefficients from the noise
                scheduler

        Returns:
            (torch.Tensor) x_new: The single-step denoised sample with the same shape as
                x.
        """
        assert isinstance(
            t, torch.Tensor
        ), f"t must be of type torch.Tensor, not {type(t)}"

        t = self.time_embedding.get_embedding(t)
        model_input = torch.cat([x, t], dim=1)

        # Compute predicted noise for the sample and subtract noise
        eps = self.model(model_input)
        x_new = subtract_scaled_noise(x, eps, alphas, alphas_tilde)

        # # TODO
        # # Use property prediction model to estimate property gradient (if provided)
        # if self.property_model is not None:
        #     property_grad = self.property_model(model_input)
        #     # TODO: Modify denoise step based on property gradient

        return x_new

    def denoise_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Given some tensor x drawn from the latent distribution, denoise it using
        the model and associated noise scheduler.

        Arguments:
            (torch.Tensor) x: The sample tensor to denoise

        Returns:
            (torch.Tensor) x_new: The denoised sample with the same shape as x.
        """
        # Iterate automatically over the noise scheduler, in reverse order
        for t, beta, alpha, alpha_tilde in reversed(list(self.noise_scheduler)):
            # Steps go from t --> t-1, so skip t = 0
            if t != 0:
                x = self._single_denoise_step(x, t, alpha, alpha_tilde)

            # Add noise (denoising step draws from new distribution) if not last step
            if t > 1:
                eps = torch.randn_like(x)
                x = x + torch.sqrt(beta) * eps

    def _train_single_epoch(self, dataloader, optimizer, criterion, device) -> float:
        """Takes a single training step where ???? TODO Complete docstring"""
        tmp_loss = []
        for batch_idx, batch in enumerate(dataloader):
            # Sample time points uniformly from the noise scheduler and get embedding
            t = torch.randint(
                low=0,
                high=self.noise_scheduler.T,
                size=(batch.shape[0],),
                device=device,
            )
            t_emb = self.time_embedding.get_embedding(t).to(device)

            # Add noise to the batch data
            batch_noised, eps = self.noise_scheduler.add_noise_to_sample(batch, t)
            model_input = torch.cat([batch_noised, t_emb], dim=1).float()

            # Predict the noise from the held model
            pred_eps = self(model_input)

            # Compute loss and take gradient step
            optimizer.zero_grad()
            loss = criterion(pred_eps, eps)
            loss.backward()
            optimizer.step()

            # Append mini-batch loss to tracked losses
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

        loss = metrics_dict["loss"]
        print(f"Finished epoch {epoch + 1} / {num_epochs}\t", end="")
        print(f"Loss: {loss:.6f}")

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
            "loss": metrics_dict["loss"],
        }
        torch.save(state_dict, checkpoint_path)

    def fit(
        self,
        dataset: LatentAndPropertyDataset,
        batch_size: int,
        num_epochs: int,
        shuffle_dl: bool,
        logging_iterations: int,
        checkpoint_iterations: int,
        optimizer_cls: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        criterion_cls: torch.nn.modules.loss._Loss,
        outdir: str,
        log_to_tensorboard: bool = True,
    ) -> None:
        """Trains the diffusion model based on the held noise scheduler. Model predicts
        added noise to the sample according to: (TODO: add paper / equations)

        Every logging_iterations epochs, statistics about training are printed to the
        console, and if a SummaryWriter object is provided, the metrics are logged to
        tensorboard.
        Every checkpoint_iterations epochs, the model weights, optimizer state, and some
        associated statistics are saved to a checkpoint file.

        Most of the attributes are read in form a configuration file, then passed to
        to this function as arguments.

        Attributes:
            (LatentAndPropertyDataset) dataset: The dataset to train on
            (int) batch_size: The batch size to use during training
            (int) num_epochs: The total number of epochs to train the model
            (bool) shuffle_dl: Whether to shuffle the dataloader
            (int) logging_iterations: How often to print out training information
            (int) checkpoint_iterations: How often to save model checkpoints
            (torch.optim.Optimizer) optimizer_cls: The optimizer class to use
            (dict) optimizer_kwargs: The keyword arguments to pass to the optimizer
                during instantiation
            (torch.nn.modules.loss._Loss) criterion_cls: The loss function class to use
            (str) outdir: The directory to save model checkpoints and tensorboard logs
                to. If None, no checkpoints or logs are saved.
            (bool) log_to_tensorboard: An optional bool specifying whether to log
                metrics to tensorboard. Default is True.

        Returns:
            None
        """
        self.train()
        device = next(self.parameters()).device
        writer = SummaryWriter(log_dir=outdir) if log_to_tensorboard else None

        # Optimizer from config attributes
        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        criterion = criterion_cls()

        # Create dataloader object from dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Start the training loop
        print(f"Starting training loop for {num_epochs} epochs")
        for epoch in range(num_epochs):
            loss = self._train_single_epoch(dataloader, optimizer, criterion, device)

            # Construct metrics dictionary
            # TODO: Add other metrics
            metrics_dict = {"loss": loss}

            # Logging and checkpointing functions internally handle intervals
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
            "loss": loss,
        }
        torch.save(state_dict, final_model_path)
