"""
This python script accepts in arguments for calculation of chemical properties of some
dataset along with the latent representations and/or the fingerprint vectors of the
molecules. Pairwise properties and latent/fingerprint vectors are calculated and saved
as tensors.

Actual datasets are instantiated as LatentAndPropertyDataset or
FingerprintAndPropertyDataset objects and saved as .pt files ready to be loaded back
into parent classes during training/inference.
"""

import os
import yaml
import argparse
from typing import List, Tuple
from pathlib import Path

import torch
from rdkit import Chem
import numpy as np

from cslvae.nn import CSLVAE
from cslvae.data import PackedTorchMol, TorchMol
from cslvae.dataset import CSLDataset

from latent_diffusion.utils import calculate_logP
from latent_diffusion.utils import calculate_QED
from latent_diffusion.utils import calculate_arbitrary_props
from latent_diffusion.utils import array_iter_batch
from latent_diffusion.dataset import LatentAndPropertyDataset
from latent_diffusion.dataset import FingerprintAndPropertyDataset

# Turn off rdkit loggging
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def parse_arguments() -> argparse.ArgumentParser:
    """Parses the command line arguments for the script and returns an ArgumentParser
    object
    """
    parser = argparse.ArgumentParser(
        prog="property_calculator",
        description="Calculate latent vector and/or fingerprint of molecules along with requested properties of a CSL dataset",
    )
    ### Items for the CSL dataset ###
    parser.add_argument(
        "--reaction_smarts_path",
        type=str,
        action="store",
        required=True,
        help="Path to reaction SMARTS file",
    )
    parser.add_argument(
        "--synthon_smiles_path",
        type=str,
        action="store",
        required=True,
        help="Path to synthon SMILES file",
    )

    ### Items for the VAE model ###
    parser.add_argument(
        "--model_weights_path",
        type=str,
        action="store",
        required=True,
        help="Path to existing model weights",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        action="store",
        required=True,
        help="Path to model configuration YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        action="store",
        required=False,
        default="",
        help="Device used for model evaluation. Defaults to cuda if available.",
    )

    ### Items for property calculation ###
    # TODO: Allow for arbitrary property calculations?
    parser.add_argument(
        "--num_molecules",
        type=int,
        action="store",
        required=True,
        help="Number of molecules to sample from the library",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        action="store",
        required=False,
        default=512,
        help="Batch size for latent variable and property prediction",
    )
    parser.add_argument(
        "--logP",
        action="store_true",
        help="Calculate logP of molecules.",
    )
    parser.add_argument(
        "--qed",
        action="store_true",
        help="Calculate QED of molecules.",
    )

    ### Items for fingerprint calculation ###
    # TODO

    ### Items for exporting and saving calculated datasets ###
    parser.add_argument(
        "--output_dir",
        type=str,
        action="store",
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        action="store",
        required=True,
        help="Name of the run. Used in constructing output directories.",
    )

    return parser.parse_args()


def parse_config(config_path: str) -> dict:
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def setup_CSLVAE(
    args: argparse.ArgumentParser, config: dict
) -> Tuple[CSLVAE, CSLDataset]:
    """Setup the CSLVAE model and dataset objects for latent variable calculations
    calculation. Returns an ordered tuple of (model, dataset).
    """
    config_model = config.get(
        "model", dict()
    )  # Default to empty dictionary if key not present

    # Load in the CSL dataset
    dataset = CSLDataset(
        reaction_smarts_path=args.reaction_smarts_path,
        synthon_smiles_path=args.synthon_smiles_path,
    )

    print(f"Loaded combinatorial synthesis library.")
    print(f"Number of reactions: {dataset.num_reactions:,}.")
    print(f"Number of synthons: {dataset.num_synthons:,}.")
    print(f"Number of products: {len(dataset):,}.")
    print()

    # Get the requested device, defaulting to cuda if available
    if len(args.device) > 0:
        device = str(args.device)
    else:
        if torch.cuda.device_count() > 0:
            device = "cuda:0"
        else:
            device = "cpu"

    # Load the model
    cslvae = CSLVAE(**config_model).to(device)

    print(f"Loading model from {args.model_weights_path}.")
    checkpoint_state_dict = torch.load(args.model_weights_path, map_location=device)
    cslvae.load_state_dict(checkpoint_state_dict["model_state_dict"])

    print("Architecture:")
    print(cslvae)
    print(f"Parameter count: {sum(p.numel() for p in cslvae.parameters()):,}.")

    return cslvae, dataset, device


def setup_output_directory(args: argparse.ArgumentParser, config: dict) -> str:
    """Creates necessary directories for outputs along with exporting information about
    the run.
    """
    outdir = Path(args.output_dir) / args.run_name

    # Make directory failing on pre-existence as to not overwrite existing data
    outdir.mkdir(parents=True, exist_ok=False)

    # Export the arguments used to run the script
    with open(outdir / "config.yaml", "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    # Export the arguments used to run the script
    args_dict = vars(args)
    with open(outdir / "args.yaml", "w") as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)

    print(f"Run Name: {args.run_name}.")
    print(f"All outputs will be written to: {outdir}.")

    return str(outdir)


def _calc_prop_tensor(mols: List[Chem.Mol], logP: bool, qed: bool) -> torch.Tensor:
    """Calculates the properties of a list of molecules and returns a tensor of shape
    (n, P_d) where n is the number of molecules and P_d is the dimension of the
    properties.

    TODO: Allow for arbitrary number of properties
    TODO: Improve logic to reduce code branching and code duplication

    Arguments:
        mols (List[Chem.Mol]): List of RDKit Mol objects of the molecules.
        logP (bool): Whether to calculate the logP of the molecule.
        qed (bool): Whether to calculate the QED of the molecule.

    Returns:
        (torch.Tensor) props: Tensor of shape (n, P_d) where n is the number of
            molecules and P_d is the dimension of the properties.
    """
    if logP and qed:
        _logP, _qed = zip(
            *[
                calculate_arbitrary_props(mol, [calculate_logP, calculate_QED])
                for mol in mols
            ]
        )
        _props = [_logP, _qed]
    else:
        if logP:
            _props = [[calculate_logP(mol)] for mol in mols]
        if qed:
            _props = [[calculate_QED(mol)] for mol in mols]

    return torch.transpose(torch.Tensor(_props), 0, 1)


def calculate_vectors_and_properties(
    cslvae: CSLVAE,
    dataset: CSLDataset,
    num_molecules: int,
    batch_size: int,
    logP: bool,
    qed: bool,
    device,
    seed: int = None,
    batch_print_interval: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Calculates the latent vectors and properties of a random subset of molecules in
    the dataset and returns a list of tuples of tensors
    (latent_vector, property_vector).

    Arguments: TODO

    Returns: TODO

    """
    total_batches = num_molecules // batch_size

    # Allocate space for tensors
    # TODO: Allow for arbitrary number of properties
    prop_dim = 2 if logP and qed else 1

    props = torch.zeros((num_molecules, prop_dim), device=device)
    z = torch.zeros((num_molecules, cslvae.query_dim), device=device)
    smiles = []

    # Set the random seed
    rng = np.random.default_rng(seed)

    # Sample indexes of the products from the dataset
    # NOTE: Not adjusted for relative number of products per reaction in the CSL
    idxs = rng.choice(len(dataset), size=num_molecules, replace=False)

    with torch.no_grad():
        for batch_num, (tensor_idx, prod_idx) in enumerate(
            array_iter_batch(arr=idxs, batch_size=batch_size)
        ):
            items = [dataset[idx] for idx in prod_idx]
            mols = [item["product"].mol for item in items]
            smiles.extend([item["product"].smiles for item in items])
            packed_mols = PackedTorchMol([item["product"] for item in items])

            # Calculate the latent vectors
            _z = cslvae.encode_molecules(packed_mols).to(device)
            z[tensor_idx] = _z

            # Calculate properties of the molecule
            _prop = _calc_prop_tensor(mols, logP, qed)
            props[tensor_idx] = torch.Tensor(_prop).to(device)

            # Print progress at each interval
            if batch_num % batch_print_interval == 0:
                print("Finished calculating batch:", batch_num, "/", total_batches)

    return z, props, smiles


def main():
    """Main function call"""
    args = parse_arguments()
    config = parse_config(args.model_config_path)

    # Verify at least one property is being calculated
    if not args.logP and not args.qed:
        raise ValueError("At least one of logP or qed must be True.")

    # Setup the output directory
    outdir = setup_output_directory(args, config)

    # Setup the model and dataset
    cslvae, dataset, device = setup_CSLVAE(args, config)

    # # Setup arguments for property calculator
    # property_functions = []
    # if args.logP:
    #     property_functions.append(calculate_logP)
    # if args.qed:
    #     property_functions.append(calculate_QED)
    # TODO: Allow other molecule properties here

    # Iterate over the dataset and calculate the properties
    z, prop, smiles = calculate_vectors_and_properties(
        cslvae=cslvae,
        dataset=dataset,
        num_molecules=args.num_molecules,
        batch_size=args.batch_size,
        logP=args.logP,
        qed=args.qed,
        device=device,
    )

    dataset = LatentAndPropertyDataset(
        latent_vectors=z, target_properties=prop, smiles=smiles
    )
    
    print("Finished calculating properties.")

    # Save the dataset
    torch.save(dataset, os.path.join(outdir, "dataset.pt"))

    print("Saved dataset to:", os.path.join(outdir, "dataset.pt"))

if __name__ == "__main__":
    main()
    print("Done!")
