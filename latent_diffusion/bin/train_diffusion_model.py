"""
Script accepts in CLI arguments (dataset, model information, training hyperparameters,
etc.) for training a diffusion model on the latent space of the CSLVAE.
"""


import argparse
import yaml


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
        default=None,
        help="Number of samples to use from the dataset. If None, use all samples.",
    )


def parse_config(config_path: str) -> dict:
    """Parses a YAML configuration file for defining model architecture and training
    variables. Returns a dictionary of the parsed configuration.
    """
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def main():
    """Main function"""
    # Parse the command line arguments
    args = parse_arguments()

    # Parse the configuration file
    config = parse_config(args.config)

    # Load pre-trained weights, if provided
    if args.diffusion_model_weights_path is not None:
        # TODO: Complete
        pass

    # Choose the device to train on
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load the dataset
    dataset = torch.load(args.dataset_path)
    dataset = dataset.float()  # Manual cast to float tensor

    # Create the time embedding
    # TODO: move logic to a separate function
    te_config = config["time_embedding"]
    te_dim = te_config["te_dim"]
    te_class = te_config["embedding_type"]
    
    # TODO: Remove if statement hardcoding to allow for more embedding types
    if te_class == "identity":
        time_embedding = IdentityTimeEmbedding(embedding_dim=te_dim)
    elif te_class == "sinusoidal":
        time_embedding = SinusoidalTimeEmbedding(embedding_dim=te_dim)
    else:
        raise ValueError(
            f"Unrecognized name for time embedding: {config['time_embedding']}"
        )

    # Create the noise scheduler
    # TODO: move logic to a separate function
    ns_config = config["noise_scheduler"]
    ns_class = ns_config["scheduler_type"]

    if ns_class == "linear":
        noise_scheduler = LinearNoiseScheduler(
            beta_0=ns_config["beta_0"],
            beta_T=ns_config["beta_T"],
            T=ns_config["T"],
            device=device,
        )
    else:
        raise ValueError(
            f"Unrecognized name for noise scheduler: {config['noise_scheduler']}"
        )

    # Create the diffusion model
    activation_cls = getattr(torch.nn, config["activation"])
    activation_kwargs = config.get("activation_kwargs", {})
    activation = activation_cls(**activation_kwargs)
    diffusion_model = PropertyGuidedDDPM(
        latent_dim=config["latent_dim"],
        te_dim=config["te_dim"],
        num_layers=config["num_layers"],
        hidden_dims=config["hidden_dims"],
        time_embedding=time_embedding,
        noise_scheduler=noise_scheduler,
        activation=activation,
    )
