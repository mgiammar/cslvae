Tasks to complete

# CSLVAE
- Implement functions to profile how accurate the VAE is at reproducing the correct molecule from some embedding.
- Implement function to screen how chemically similar (e.g. from fingerprints) the nearest neighbors / region in latent space is when decoded.
- Verify fitting to custom CSL is working, and improve tracked metrics.

# Latent Diffusion
-  Move parsing functionality from trining scripts to class methods to reduce code duplication and improve flexibility. This would include parsing nested objects (e.g. noise schedules). For example:
```
class PropertyGuidedDDPM(nn.Module):

    @classmethod
    def parse_config(config: dict) -> PropertyGuidedDDPM:
        # Parse options in config dictionary, then call __init__
        ...

    def __init__(self, **kwargs):
        # Instantiation logic here
        ...
```
- Add/improve datatype cascading and validation throughout code. Infer data types from model and/or dataset?
- Implement more robust validation checks at the class level. For example, make sure the noise scheduler objects between a `PropertyGuidedDDPM` and `TimeDependentPropertyModel` match before sampling

# Documentation / Presentations
- Make better figure showing how data is being passed through the model. Include specific model architectures.
- Make figure showing how training loop and losses are being calculated including aspects of architecture figure.
- UML diagram for latent diffusion objects
- Diagram showing how to train/run a property guided latent diffusion model.