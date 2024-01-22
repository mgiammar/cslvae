"""
Script accepts in CLI arguments (dataset, model information, training hyperparameters,
etc.) for training a property predictive model from the latent spae of the CSLVAE.
Currently, the only supported architecture is a MLP model either with time-dependence
(for working alongside a diffusion model for property-guided sampling) or just as a
simple model for predicting p(y|z).
"""
