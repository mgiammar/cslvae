"""
Useful utility function for property calculation and latent diffusion.
"""

from typing import List, Dict, Tuple, Union, Callable, Iterable

import numpy as np

import torch

import rdkit
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed


def calculate_logP(mol: Chem.Mol) -> float:
    """Calculate the logP of a molecule using the Crippen method
    (https://doi.org/10.1021/ci990307l).

    Args:
        mol (Chem.Mol): RDKit Mol object of the molecule

    Returns:
        float: logP of the molecule
    """
    return MolLogP(mol)


def calculate_QED(mol: Chem.Mol) -> float:
    """Calculate the QED of a molecule using the QED method
    (https://doi.org/10.1038/nchem.1243).

    Args:
        mol (Chem.Mol): RDKit Mol object of the molecule

    Returns:
        float: QED of the molecule
    """
    return qed(mol)


def calculate_arbitrary_props(
    mol: Chem.Mol,
    prop_functions: List[Callable[[Chem.Mol], Union[float, List[float]]]],
    prop_function_kwargs: List[Dict] = None,
) -> List[float]:
    """Calculate the arbitrary properties of a molecule using the QED method
    (https://doi.org/10.1038/nchem.1243).

    Arguments:
        mol (Chem.Mol): RDKit Mol object of the molecule
        prop_functions (List[Callable[[Chem.Mol], float]]): List of functions to call
            on the molecule which return a float value(s) (e.g. CalcPhi(mol)). Functions
            which return a list of values will be flattened out into a 1D list.
        prop_function_kwargs (List[Dict]): Ordered list of dict objects of keyword
            arguments to pass to the property function calls.

    Returns:
        (float) props: Calculated properties of the molecule flattened out into a 1D.
    """

    def flatten(iterable):
        """Takes in a nested list structure and flattens it out into a 1D list.

        Example:
            >>> foo = [1, 2, [3, 4], [5, 6, 7], 8]
            >>> list(flatten(foo))
            [1, 2, 3, 4, 5, 6, 7, 8]
        """
        for element in iterable:
            if isinstance(element, Iterable):
                yield from flatten(element)
            else:
                yield element

    if prop_function_kwargs is None:
        prop_function_kwargs = [{}] * len(prop_functions)

    props = [
        fn(mol, **kwargs) for fn, kwargs in zip(prop_functions, prop_function_kwargs)
    ]
    props = list(flatten(props))

    return props


def array_iter_batch(
    arr: np.ndarray, batch_size: int
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Iterates over an array in batches of size batch_size. Yields the indices of the
    current batch (e.g. [0, 1, ... batch_size - 1]) and the batch itself.
    """
    current_idx = 0
    while current_idx < arr.size:
        yield (
            np.arange(current_idx, min(current_idx + batch_size, arr.size)),
            arr[current_idx : current_idx + batch_size],
        )
        current_idx += batch_size


def subtract_scaled_noise(
    x: torch.Tensor,
    eps: torch.tensor,
    alphas: torch.Tensor,
    alphas_tilde: torch.Tensor,
) -> torch.Tensor:
    """Utility function for denoising diffusion models to subtract already predicted
    noise (eps) from a sample (x) using the proper coefficients (alphas, alphas_tilde)
    from some noise scheduler class. This can perform batched denoising.

    Arguments:
        (torch.Tensor) x: The sample tensor to denoise
        (torch.Tensor) eps: The predicted noise on x (computed from external model)
        (torch.Tensor) alphas: The alphas coefficients from the noise scheduler
        (torch.Tensor) alphas_tilde: The alphas_tilde coefficients from the noise
            scheduler

    Returns:
        (torch.Tensor) x_new: The single-step denoised sample
    """
    # TODO: Move this into the sampling loop (unncessary to have util function)
    eps = eps * (1 - alphas)
    eps = eps / torch.sqrt(1 - alphas_tilde)
    x_new = x - eps
    x_new = x_new / torch.sqrt(alphas)

    return x_new


def scale_betas_for_zero_snr(betas: torch.Tensor) -> torch.Tensor:
    """Implements the method in https://arxiv.org/pdf/2305.08891.pdf for scaling beta
    values to achieve a zero SNR at t=T for some noise schedule.

    Arguments:
        (torch.Tensor) betas: The beta values to scale

    Returns:
        (torch.Tensor) betas_scaled: The scaled beta values
    """
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = torch.sqrt(alphas_bar)

    # Store old values
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so last timestep has value of zero
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale linearly such that first timestep remains the same
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert back to beta values
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[:1] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas