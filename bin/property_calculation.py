import yaml
import argparse
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from cslvae.dataset import CSLDataset
from cslvae.nn import CSLVAE, CSLVAEDB
from cslvae.data import PackedTorchMol, TorchMol


def parse_cli_arguments():
    """Parse in arguments for this script and return as a dictionary."""
    parser = argparse.ArgumentParser(
        description="Run CSLVAE training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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

    parser.add_argument(
        "--config",
        type=str,
        action="store",
        required=True,
        help="Path to YAML file specifying the configuration.",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        action="store",
        required=True,
        help="Path to existing model weights.",
    )

    parser.add_argument(
        "--num_products",
        type=int,
        required=True,
        help="Number of products in the library to generate.",
    )

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
        help="The name of the property prediction run.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        action="store",
        required=False,
        default=512,
        help="The name of the property prediction run.",
    )

    return parser.parse_args()


def parse_config_from_args(args):
    """Parses the YAML config file from the provided arguments."""
    with open(args.config, "r") as fp:
        config = yaml.load(fp, Loader=yaml.CLoader)

    return config


def get_dataset_and_model(args, config):
    # Choose device
    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"

    # Get config parts
    config_model = config.get("model", dict())
    config_train = config.get("training", dict())

    # Load in CSL dataset
    dataset = CSLDataset(args.reaction_smarts_path, args.synthon_smiles_path)

    # Load in model
    cslvae = CSLVAE(**config_model).to(device)
    print(f"Loading model from {args.weights_path}.")
    checkpoint_state_dict = torch.load(args.weights_path, map_location=device)
    cslvae.load_state_dict(checkpoint_state_dict["model_state_dict"])

    return dataset, cslvae


def sample_n_products_indexes_reaction_adjusted(
    dataset, num_products: int, seed: int = None
) -> np.ndarray:
    """Given a dataset and desired number of samples, return the indexes of the
    uniformly sampled products from each reaction. Since each reaction can produce
    different numbers of products, sampling is weighted so products are uniform w.r.t.
    reaction index.

    Args:
        dataset (CSLDataset): The dataset to sample from.
        num_products (int): The number of products to sample.
        seed (int, optional): The random seed to use. Defaults to None.

    Returns:
        np.ndarray: The indexes of the sampled products.
    """
    p = dataset._reaction_counts.max() / dataset._reaction_counts
    p /= p.sum()

    rng = np.random.default_rng(seed)
    reaction_indexes = rng.choice(
        np.arange(len(dataset._reaction_counts)), size=num_products, p=p
    )

    return reaction_indexes


def array_iter_batch(arr, batch_size):
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


if __name__ == "__main__":
    args = parse_cli_arguments()
    config = parse_config_from_args(args)

    # Create output directories, start logs, and save a copy of the config file (in case needed as
    # reference)
    run_name = args.run_name
    outdir = os.path.join(args.output_dir, run_id)
    os.makedirs(os.path.join(outdir, "checkpoints"))
    shutil.copy(args.config, os.path.join(outdir, "config.yaml"))

    ### =========================================================================== ###
    ###                     Setting up model and dataset object                     ###
    ### =========================================================================== ###
    # Choose device
    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"

    # Output some information about the run
    print(f"Run ID: {run_name}.")
    print(f"All outputs will be written to: {outdir}.")
    print(f"GPU count: {torch.cuda.device_count()}. CPU count: {os.cpu_count()}.")
    print(f"Training on device: {device}.")
    print()

    dataset, cslvae = get_dataset_and_model(args, config)

    print(f"Loaded combinatorial synthesis library.")
    print(f"Number of reactions: {dataset.num_reactions:,}.")
    print(f"Number of synthons: {dataset.num_synthons:,}.")
    print(f"Number of products: {len(dataset):,}.")
    print()

    print("Model Architecture:")
    print(cslvae)
    print(f"Parameter count: {sum(p.numel() for p in cslvae.parameters()):,}.")
    print()

    ### =========================================================================== ###
    ###                     Sampling Products from CSL by Index                     ###
    ### =========================================================================== ###
    # # non-reaction adjusted sampling
    # product_indexes = rng.choice(
    #     np.arange(len(dataset._reaction_counts)), size=args.num_products, p=p
    # )

    # reaction adjusted sampling
    product_indexes = sample_n_products_indexes_reaction_adjusted(
        dataset, args.num_products
    )

    ### =========================================================================== ###
    ###                     Calculating latent z and LogP values                    ###
    ### =========================================================================== ###
    # TODO: Move this to a single function, maybe using partial statements to pass in
    # arbitrary property prediction function.
    batch_size = args.batch_size

    # Allocate space for tensors
    z = torch.zeros(
        (args.num_products, cslvae.latent_dim), device=device, requires_grad=False
    )
    logP = torch.zeros((args.num_products,), device=device, requires_grad=False)

    cslvae.eval()
    with torch.no_grad():
        for batch_num, (tensor_idxs, prod_idxs) in enumerate(
            array_iter_batch(arr=product_indexes, batch_size=batch_size)
        ):
            items = [dataset[i] for i in prod_idxs]

            # Get PackedTorchMol object from the chunk of products
            packed_mols = PackedTorchMol([item["product"] for item in items])
            mols = [item["product"].mol for item in items]

            # Get latent space representation for the batch
            _z = cslvae.encode_molecules(packed_mols)
            _logP = torch.Tensor([MolLogP(mol) for mol in mols])

            # Store the latent space representation
            z[tensor_idxs] = _z
            logP[tensor_idxs] = _logP

            if batch_num % 10 == 0:
                print(
                    "Finished calculating batch:",
                    batch_num,
                    "/",
                    len(product_indexes // batch_size),
                )

    torch.save(z, os.path.join(outdir, "z.pt"))
    torch.save(z, os.path.join(outdir, "logP.pt"))

    print("Tensors saved to disk.")
    ### =========================================================================== ###
