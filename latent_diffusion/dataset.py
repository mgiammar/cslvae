"""
Important classes for dealing with property and vector representations (latent dimension
of VAE or fingerprints) of molecules. TODO: Add more documentation.
"""

from typing import List

import torch
from torch.utils.data import Dataset, TensorDataset


class LatentAndPropertyDataset(TensorDataset):
    """Class which holds the latent representation of a molecule (feature) and some
    property (label/target) also associated with the molecule. Also holds an optional
    SMILES string for each molecule.

    Attributes:
        (torch.Tensor) latent_vectors: Tensor of shape (n, Z_d) where n is the number
            of elements in the dataset and Z_d is the dimension of the latent space.
        (torch.Tensor) target_properties: Tensor of shape (n, P_d) where n is the is the
            number of elements in the dataset and P_d is the dimension of the
                property/properties.
        (List[str]) smiles: Optional list of SMILES strings for each molecule. Default
            is None.
    """

    def __init__(
        self,
        latent_vectors: torch.Tensor,
        target_properties: torch.Tensor,
        smiles: List[str] = None,
    ):
        self.smiles = smiles
        super().__init__(latent_vectors, target_properties)


class FingerprintAndPropertyDataset(TensorDataset):
    """Class which holds calculated fingerprint of a molecule (feature) and some
    property (label/target) also associated with the molecule. This is similar to
    the LatentAndPropertyDataset, but has additional metadata w.r.t. the kind of
    fingerprint calculated form the molecules. Also holds an optional
    SMILES string for each molecule.


      Attributes:
        (torch.Tensor) fingerprints: Tensor of shape (n, F_d) where n is the number
            of elements in the dataset and F_d is the dimension of the fingerprint.
        (torch.Tensor) target_properties: Tensor of shape (n, P_d) where n is the is the
            number of elements in the dataset and P_d is the dimension of the
                property/properties.
        (List[str]) smiles: Optional list of SMILES strings for each molecule. Default
            is None.

    TODO: Complete

    """

    def __init__(
        self,
        fingerprints: torch.Tensor,
        target_properties: torch.Tensor,
        fingerprint_type: str,
        smiles: List[str] = None,
    ):
        self.fingerprint_type = fingerprint_type
        self.smiles = smiles
        super().__init__(fingerprints, target_properties)
