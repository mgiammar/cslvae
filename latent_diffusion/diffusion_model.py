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

    def denoise_sample(self, x):
        """Given some tensor x drawn from the latent distribution, denoise it using
        the model and associated noise scheduler.

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
        """Takes a single training step where ????"""
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

            # print("t            \t:", t.dtype)
            # print("t_emb        \t:", t_emb.dtype)
            # print("batch        \t:", batch.dtype)
            # print("batch_noised \t:", batch_noised.dtype)
            # print("eps          \t:", eps.dtype)
            # print("model_input  \t:", model_input.dtype)
            # print("pred_eps     \t:", pred_eps.dtype)

            # Compute loss and take gradient step
            optimizer.zero_grad()
            loss = criterion(pred_eps, eps)
            loss.backward()
            optimizer.step()

            # Append mini-batch loss to tracked losses
            tmp_loss.append(loss.item())

        return np.mean(tmp_loss)

    def _train_logging_function(self, epoch, num_epochs, metrics_dict, writer):
        """TODO: Docstring"""
        loss = metrics_dict["loss"]
        print(f"Finished epoch {epoch + 1} / {num_epochs}\t", end="")
        print(f"Loss: {loss:.6f}")

        # Log metrics to tensorboard, if requested
        if writer is not None:
            for metric_key, metric_value in metrics_dict.items():
                writer.add_scalar(f"{metric_key}", metric_value, epoch)

    def _train_checkpoint_function(
        self, epoch, optimizer, metrics_dict, checkpoint_path
    ):
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
        config: dict,
        outdir: str = None,
    ):
        """Trains the diffusion model based on the held noise scheduler. Model predicts
        added noise to the sample according to: (TODO: add paper / equations)

        Attributes: TODO
        """
        device = next(self.parameters()).device
        writer = None if outdir is None else SummaryWriter(log_dir=outdir)

        # Get training hyperparameters from config dictionary
        batch_size = config["batch_size"]
        num_epochs = config["num_epochs"]
        shuffle_dl = config["shuffle_dl"]
        logging_iterations = config["logging_iterations"]
        checkpoint_iterations = config["checkpoint_iterations"]

        # Optimizer from config attributes
        opt_class_name = config["opt_class_name"]
        opt_class = getattr(torch.optim, opt_class_name)
        opt_kwargs = config["opt_kwargs"]
        optimizer = opt_class(self.parameters(), **opt_kwargs)
        criterion = nn.MSELoss()  # NOTE: This is fixed

        # Create dataloader object from dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Start the training loop
        print(f"Starting training loop for {num_epochs} epochs")
        for epoch in range(num_epochs):
            loss = self._train_single_epoch(dataloader, optimizer, criterion, device)

            # Construct metrics dictionary
            metrics_dict = {
                "loss": loss,
                # TODO: Add other metrics
            }

            # Print out training information every logging_iterations epochs
            if (epoch % logging_iterations == 0) or (epoch == num_epochs - 1):
                self._train_logging_function(epoch, num_epochs, metrics_dict, writer)

            # Save model checkpoint every checkpoint_iterations epochs
            if (
                (outdir is not None)
                and (epoch % checkpoint_iterations == 0)
                or (epoch == num_epochs - 1)
            ):
                checkpoint_path = os.path.join(outdir, f"model_checkpoint_{epoch}.pth")
                self._train_checkpoint_function(
                    epoch, optimizer, metrics_dict, checkpoint_path
                )

        if writer is not None:
            writer.close()

        # Save final model
        if outdir is not None:
            final_model_path = os.path.join(outdir, "model_checkpoint_final.pth")
            state_dict = {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
            torch.save(state_dict, final_model_path)
