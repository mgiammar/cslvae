"""
Script accepts in CLI arguments (dataset, model information, training hyperparameters,
etc.) for training a property predictive model from the latent spae of the CSLVAE.
Currently, the only supported architecture is a MLP model either with time-dependence
(for working alongside a diffusion model for property-guided sampling) or just as a
simple model for predicting p(y|z).
"""


import argparse
import yaml

# from latent_diffusion.


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

    # Property model config file
    parser.add_argument(
        "--property_model_config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the property model.",
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
        "--property_model_weights_path",
        type=str,
        required=False,
        default=None,
        help="Path to the pre-trained weights for the property model.",
    )

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
    outdir = Path(args.output_dir) / args.run_name

    # Make directory failing on pre-existence as to not overwrite existing data
    outdir.mkdir(parents=True, exist_ok=False)

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
    te_dim = te_config["te_dim"]
    te_class = te_config["embedding_type"]

    # TODO: Remove if statement hardcoding to allow for more embedding types
    if te_class == "identity":
        time_embedding = IdentityTimeEmbedding(embedding_dim=te_dim)
    elif te_class == "sinusoidal":
        time_embedding = SinusoidalTimeEmbedding(embedding_dim=te_dim)
    else:
        raise ValueError(f"Unrecognized name for time embedding: {te_class}")

    return time_embedding


def get_noise_scheduler_object(config: dict) -> DiscreteNoiseScheduler:
    # Create the noise scheduler
    ns_config = config["noise_scheduler"]
    ns_class = ns_config["scheduler_type"]
    beta_0 = ns_config["beta_0"]
    beta_T = ns_config["beta_T"]
    T = ns_config["T"]

    if ns_class == "linear":
        noise_scheduler = LinearNoiseScheduler(beta_0, beta_T, T)
    else:
        raise ValueError(f"Unrecognized name for noise scheduler: {ns_class}")

    return noise_scheduler


def get_property_model_object(
    config: dict,
    time_embedding: TimeEmbedding = None,
    noise_scheduler: DiscreteNoiseScheduler = None,
) -> nn.Module:
    """Constructs a property model from the provided configuration dictionary."""
    # Get values from the property model config portion of the config dictionary
    pm_config = config["property_model"]
    pm_cls_name = pm_config.pop("class_name")

    # Add the time embedding and noise scheduler to the property model config, if
    # model is a time-dependent model
    if pm_cls_name == "TTimeDependentPropertyModel":
        pm_config["time_embedding"] = time_embedding
        pm_config["noise_scheduler"] = noise_scheduler

    # Instantiate instance the property model class with desired arguments
    pm_cls = getattr(latent_diffusion.property_model, pm_cls_name)
    property_model = pm_cls(**pm_config)

    return property_model


def construct_fit_kwargs(config: dict) -> dict:
    """TODO docstring"""
    # NOTE: dict.get method is employed for some args to allow for default values
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    logging_iterations = config.get("logging_iterations", 20)
    checkpoint_iterations = config.get("checkpoint_iterations", 100)

    shuffle_train_dl = config.get("shuffle_dl", True)
    shuffle_test_dl = config.get("shuffle_dl", False)

    optimizer_cls = getattr(torch.optim, config["opt_class_name"])
    optimizer_kwargs = config.get("opt_kwargs", {})

    criterion_cls = getattr(torch.nn, config["criterion_class_name"])

    train_test_split = config.get("train_test_split", (0.8, 0.2))

    return {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "logging_iterations": logging_iterations,
        "checkpoint_iterations": checkpoint_iterations,
        "optimizer_cls": optimizer_cls,
        "optimizer_kwargs": optimizer_kwargs,
        "criterion_cls": criterion_cls,
        "shuffle_train_dl": shuffle_train_dl,
        "shuffle_test_dl": shuffle_test_dl,
        "train_test_split": train_test_split,
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
    if len(dataset) < args.num_samples:
        raise ValueError(
            f"Dataset contains {len(dataset)} samples which is fewer than the requested "
            f"num_samples ({args.num_samples})."
        )
    dataset = dataset[: args.num_samples] if args.num_samples is not None else dataset

    # Choose the device to train on
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    time_embedding = get_time_embedding_object(config)
    noise_scheduler = get_noise_scheduler_object(config)
    property_predictor_model = get_property_model_object(
        config,
        time_embedding,
        noise_scheduler,
    )
    property_predictor_model = property_predictor_model.to(device)

    # Load pre-trained weights, if provided
    # TODO: Implement
    if args.property_model_weights_path is not None:
        raise NotImplementedError

    # Parsing items from the config dictionary to pass to the fit method, then call
    # fit method
    fit_kwargs = construct_fit_kwargs(config)
    property_predictor_model.fit(
        dataset=dataset,
        outdir=outdir,
        log_to_tensorboard=args.log_to_tensorboard**fit_kwargs,
    )
