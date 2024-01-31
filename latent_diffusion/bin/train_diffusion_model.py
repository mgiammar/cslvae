"""
Script accepts in CLI arguments (dataset, model information, training hyperparameters,
etc.) for training a diffusion model on the latent space of the CSLVAE.
"""


import argparse
import yaml
from pathlib import Path

import torch

from latent_diffusion.time_embedding import TimeEmbedding
from latent_diffusion.time_embedding import IdentityTimeEmbedding
from latent_diffusion.time_embedding import SinusoidalTimeEmbedding
from latent_diffusion.noise_scheduler import DiscreteNoiseScheduler
from latent_diffusion.noise_scheduler import LinearNoiseScheduler
from latent_diffusion.diffusion_model import PropertyGuidedDDPM



def parse_arguments():
    """Parses the command line arguments fore the script and returns an ArgumentParser
    object.
    """
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on the latent space of the CSLVAE."
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset to use for training the diffusion model.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=None,
        help="Number of samples to use from the dataset. Default is to use all samples.",
    )
    parser.add_argument(
        "--normalize_dataset",
        action="store_true",
        required=False,
        default=False,
        help=(
            "Whether to normalize the dataset to have zero mean and unit variance "
            "before training diffusion model. Calculated mean and variance tensors are "
            "saved to output directory before training. Default is False."
        )
    )

    # Model configuration arguments
    parser.add_argument(
        "--diffusion_model_config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the diffusion model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        action="store",
        required=False,
        default=None,
        help="Device used for model training. Default is to use CUDA, if available.",
    )
    parser.add_argument(
        "--diffusion_model_weights_path",
        type=str,
        required=False,
        default=None,
        help="Path to the pre-trained weights for the property model.",
    )

    # Output/export arguments
    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        default=".",
        help="Path to the directory where the output files will be saved.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run. Used for naming the output files.",
    )
    #parser.add_argument(
    #    "--log_to_tensorboard",
    #    action="store_true",
    #    required=False,
    #    default=True,
    #    help="Whether to log training metrics to TensorBoard.",
    #)


    return parser.parse_args()


def parse_config(config_path: str) -> dict:
    """Parses a YAML configuration file for defining model architecture and training
    variables. Returns a dictionary of the parsed configuration.
    """
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def setup_output_directory(args: argparse.ArgumentParser, config: dict) -> str:
    """Creates necessary directories for outputs along with exporting information about
    the run.
    """
    outdir = Path(args.outdir) / args.run_name

    # Make directory failing on pre-existence as to not overwrite existing data
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=False)

    # Export the arguments used to run the script
    with open(outdir / "diffusion_model_config.yaml", "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    # Export the arguments used to run the script
    args_dict = vars(args)
    with open(outdir / "args.yaml", "w") as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)

    print(f"Run Name: {args.run_name}.")
    print(f"All outputs will be written to: {outdir}.")

    return str(outdir)


def get_time_embedding_object(config) -> TimeEmbedding:
    # Create the time embedding
    te_config = config["time_embedding"]
    te_dim = te_config["embedding_dim"]
    te_class = te_config["embedding_type"]

    # TODO: Remove if statement hardcoding to allow for more embedding types
    if te_class == "identity":
        time_embedding = IdentityTimeEmbedding(dim=te_dim)
    elif te_class == "sinusoidal":
        time_embedding = SinusoidalTimeEmbedding(dim=te_dim)
    else:
        raise ValueError(f"Unrecognized name for time embedding: {te_class}")

    return time_embedding


def get_noise_scheduler_object(config: dict) -> DiscreteNoiseScheduler:
    # Create the noise scheduler
    ns_config = config["noise_scheduler"]
    ns_class = ns_config["schedule"]
    beta_0 = float(ns_config["beta_start"])
    beta_T = float(ns_config["beta_stop"])
    T = ns_config["T"]

    if ns_class == "linear":
        noise_scheduler = LinearNoiseScheduler(beta_0, beta_T, T)
    else:
        raise ValueError(f"Unrecognized name for noise scheduler: {ns_class}")

    return noise_scheduler


def get_diffusion_model_object(
    config: dict, time_embedding, noise_scheduler
) -> PropertyGuidedDDPM:
    # Create the diffusion model
    model_config = config["diffusion_model"]

    diffusion_model = PropertyGuidedDDPM.parse_config(
        model_config
    )

    #activation_cls = getattr(torch.nn, model_config["activation"])
    #activation_kwargs = model_config.get("activation_kwargs", {})
    #activation = activation_cls(**activation_kwargs)
    #diffusion_model = PropertyGuidedDDPM(
    #    input_dim=model_config["input_dim"],
    #    time_embedding_dim=model_config["time_embedding_dim"],
    #    num_hidden_layers=model_config["hidden_layers"],
    #    hidden_layer_shapes=model_config["hidden_layer_sizes"],
    #    time_embedding=time_embedding,
    #    noise_scheduler=noise_scheduler,
    #    activation=activation,
    #)

    return diffusion_model


def construct_fit_kwargs(config: dict, outdir: str) -> dict:
    """TODO docstring"""
    # NOTE: dict.get method is employed for some args to allow for default values
    fit_config = config["training"]
    batch_size = fit_config["batch_size"]
    num_epochs = fit_config["num_epochs"]
    shuffle_dl = fit_config.get("shuffle_dl", True)
    logging_iterations = fit_config.get("logging_iterations", 20)
    checkpoint_iterations = fit_config.get("checkpoint_iterations", 100)

    # Optimizer arguments
    optimizer_cls = getattr(torch.optim, fit_config["optimizer_cls"])
    optimizer_kwargs = fit_config.get("optimizer_kwargs", {})

    # Criterion arguments
    criterion_cls = getattr(torch.nn, fit_config["criterion"])

    log_to_tensorboard = fit_config.get("log_to_tensorboard", True)

    return {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "shuffle_dl": shuffle_dl,
        "logging_iterations": logging_iterations,
        "checkpoint_iterations": checkpoint_iterations,
        "optimizer_cls": optimizer_cls,
        "optimizer_kwargs": optimizer_kwargs,
        "criterion_cls": criterion_cls,
        "outdir": outdir,
        "log_to_tensorboard": log_to_tensorboard,
    }


def main():
    """Main function"""
    # Parse the command line arguments
    args = parse_arguments()

    # Parse the configuration file
    config = parse_config(args.diffusion_model_config)

    outdir = setup_output_directory(args, config)

    # Load the dataset, then subsample to num_samples if requested
    dataset = torch.load(args.dataset_path)
    if args.num_samples is not None and len(dataset) < args.num_samples:
        raise ValueError(
            f"Dataset contains {len(dataset)} samples which is fewer than the "
            f"requested num_samples ({args.num_samples})."
        )
    dataset = dataset[: args.num_samples] if args.num_samples is not None else dataset
    if args.normalize_dataset:
        mean = dataset.mean(dim=0, keepdim=True)
        sigma = dataset.std(dim=0, keepdim=True)

        # Save the mean and sigma tensors to the output directory
        torch.save(mean, outdir + "/" + args.run_name + "_mean.pt")
        torch.save(sigma, outdir + "/" + args.run_name + "_sigma.pt")

        dataset = (dataset - mean) / sigma

    # Choose the device to train on
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # time_embedding = get_time_embedding_object(config)
    # noise_scheduler = get_noise_scheduler_object(config)
    # diffusion_model = get_diffusion_model_object(
    #    config, time_embedding, noise_scheduler
    # )
    # diffusion_model = get_diffusion_model_object(config)
    diffusion_model = PropertyGuidedDDPM.parse_config(config["diffusion_model"])
    diffusion_model = diffusion_model.to(device)

    # Load pre-trained weights, if provided
    # TODO: Implement
    if args.diffusion_model_weights_path is not None:
        raise NotImplementedError

    # Parsing items from the config dictionary to pass to the fit method, then call
    # fit method
    fit_kwargs = construct_fit_kwargs(config, outdir)
    diffusion_model.fit(
        dataset=dataset,
        **fit_kwargs,
    )


if __name__ == "__main__":
    main()
