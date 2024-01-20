"""
Useful utility function for property calculation and latent diffusion.
"""

from typing import List, Dict, Tuple, Union, Callable, Iterable

import numpy as np

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
