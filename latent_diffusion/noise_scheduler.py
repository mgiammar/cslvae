"""Module for noise scheduling classes for use in diffusion models."""

from typing import Union, Tuple
import torch
import numpy as np

from latent_diffusion.utils import scale_betas_for_zero_snr


class DiscreteNoiseScheduler:
    """Base noise scheduler class which accepts in a list of betas for for defining the
    variance of noise added to samples during the noising process.

    TODO describe implemented methods

    TODO equations

    Attributes:
        (int) T: The number of timesteps in the noise schedule
        (torch.Tensor) times: The times at which the noise schedule is defined
        (torch.Tensor) betas: The betas defining the noise schedule
        (torch.Tensor) alphas: The alpha values (calc. as 1-beta)
        (torch.Tensor) alphas_cumprod: The cumulative product of alpha values

    Methods:
        (iterator) __iter__: Utility method for iterating over times, betas, alphas, and
            alphas_cumprod held by the scheduler, in that order.
    """

    @classmethod
    def parse_config(cls, config: dict):
        beta_start = float(config["beta_start"])
        beta_stop = float(config["beta_stop"])
        T = int(config["T"])
        enforce_zero_snr = config.get("enforce_zero_snr", False)

        _type = config["type"]
        if _type == "linear":
            return LinearNoiseScheduler(beta_start, beta_stop, T, enforce_zero_snr)
        else:
            raise ValueError(f"Unrecognized noise scheduler type: {_type}")

    def __init__(self, betas: Union[list, np.ndarray, torch.Tensor]):
        # Ensure the accepted in beta values are 1-dimensional by casting to tensor
        if not isinstance(betas, (list, np.ndarray, torch.Tensor)):
            raise TypeError(
                f"betas must be list, np.ndarray, or torch.Tensor. Found {type(betas)}."
            )

        if isinstance(betas, (list, np.ndarray)):
            betas = torch.tensor(betas)

        if betas.ndim != 1:
            raise ValueError(
                f"betas must be 1-dimensional. Got {betas.ndim} dimensions."
            )

        # Setup attributes
        self.T = betas.shape[0]
        self.times = torch.arange(self.T)
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def __iter__(self):
        """Utility method for iterating over times, betas, alphas, and alphas_cumprod
        held by the scheduler.

        Returns
            (iterator): A zip iterator over (times, betas, alphas, and alphas_cumprod)
                in that order.
        """
        return zip(self.times, self.betas, self.alphas, self.alphas_cumprod)

    def to(self, device):
        self.times = self.times.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

    def add_noise_to_sample(
        self, sample: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adds noise to the input sample according to the noise schedule at the
        specified timestep.

        Arguments:
            (torch.Tensor) sample: The sample to add noise to
            (torch.Tensor) t: The timestep to add noise at

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]) (sample_noised, noise): The sample with
                added noise and the exact noise added.
        """
        # Check for valid input
        if torch.any(t < 0) or torch.any(t >= self.T):
            raise ValueError(f"t must be between 0 and {self.T}")

        # Draw random noise and scale based on coefficients
        eps = torch.randn_like(sample, device=sample.device, dtype=sample.dtype)
        at = torch.unsqueeze(self.alphas[t], dim=1)
        at = at.expand_as(sample)
        sqrt_1_minus_at = torch.sqrt(1 - at)

        # Return the noised sample and the exact added noise
        return torch.sqrt(at) * sample + sqrt_1_minus_at * eps, eps


class LinearNoiseScheduler(DiscreteNoiseScheduler):
    """Defines a linear noise schedule from beta_0 to beta_T over T timesteps. Inherits
    from NoiseScheduler so all attributes and methods from the parent class are
    accessible.

    TODO equations

    TODO Complete docstring
    """

    def __init__(
        self,
        beta_start: float,
        beta_stop: float,
        T: int,
        enforce_zero_snr: bool = False,
    ):
        self.beta_start = beta_start
        self.beta_stop = beta_stop
        betas = np.linspace(beta_start, beta_stop, T, endpoint=True, dtype=np.float32)

        if enforce_zero_snr:
            betas = scale_betas_for_zero_snr(betas)

        super().__init__(betas)


class ExponentialNoiseScheduler(DiscreteNoiseScheduler):
    pass


class CosineNoiseScheduler(DiscreteNoiseScheduler):
    pass
