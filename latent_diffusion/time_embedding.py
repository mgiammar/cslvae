"""Module for implementing a variety of time embedding methods."""


from typing import Union
from abc import ABC, abstractmethod
import torch


class TimeEmbedding(ABC):
    """Instances of this class handles creating a time embedding vector."""

    @classmethod
    def parse_config(cls, config: dict):
        """TODO: docstring"""
        dim = int(config["dim"])
        _type = config["type"]

        # TODO: Make this an arbitrary map to classes
        if _type == "identity":
            return IdentityTimeEmbedding(dim=dim)
        elif _type == "sinusoidal":
            return SinusoidalTimeEmbedding(dim=dim)
        else:
            raise ValueError(f"Unrecognized time embedding type: {_type}")

    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim

    @abstractmethod
    def get_embedding(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Given an input tensor of times, compute the embedding for those times and
        return as a tensor.

        Arguments:
            (Union[int, torch.Tensor]) t: The input time either as an integer or as a
                tensor of integers.

        Returns:
            (torch.Tensor) embedding: The time embedding for the input times.
        """
        ...


class IdentityTimeEmbedding(TimeEmbedding):
    """Embedding is identity broadcast to the number of embedding dimensions"""

    def __init__(self, dim: int):
        super().__init__(name=__class__.__name__, dim=dim)

    def get_embedding(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Implements the abstract get_embedding method where the desired time is simply
        broadcast to the number of embedding dimensions; no additional transformations
        are applied to the tensor.

        Arguments:
            (Union[int, torch.Tensor]) t: The input time

        Returns:
            (torch.Tensor) embedding: The time embedding for the input time(s).
        """
        if isinstance(t, int):
            t = torch.ones((1,), dtype=torch.int32) * t

        embedding = torch.unsqueeze(t, dim=1)
        embedding = torch.broadcast_to(embedding, size=[t.shape[0], self.dim])

        return embedding


class SinusoidalTimeEmbedding(TimeEmbedding):
    """Implements sinusoidal time embedding according to
    https://arxiv.org/pdf/1706.03762.
    """

    def __init__(self, dim: int):
        super().__init__(name=__class__.__name__, dim=dim)

    def get_embedding(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Given an input tensor of times, compute the embedding for those times and
        return as a tensor.

        Arguments:
            (Union[int, torch.Tensor]) t: The input time

        Returns:
            (torch.Tensor) embedding: The time embedding for the input time(s).
        """
        if isinstance(t, int):
            t = torch.ones((1,), dtype=torch.int32) * t

        embedding = torch.pow(10000, torch.arange(self.dim) / self.dim)

        # Shape broadcaatings for proper division
        embedding = torch.unsqueeze(embedding, dim=0)
        embedding = torch.broadcast_to(embedding, size=[t.shape[0], self.dim])
        embedding = torch.div(torch.unsqueeze(t, dim=1), embedding)

        # Call sinusoidal functions
        embedding[:, 0::2] = embedding[:, 0::2].sin()
        embedding[:, 1::2] = embedding[:, 1::2].cos()

        return embedding


class PolynomialTimeEmbedding(TimeEmbedding):
    """Implements polynomial time embedding."""

    pass
