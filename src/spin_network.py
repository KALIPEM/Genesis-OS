from typing import Iterable, List, Optional

import torch
from torch import nn

from .spin_neuron import SpinNeuron


class SpinNetwork(nn.Module):
    """Feed-forward network composed of :class:`SpinNeuron` layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: Optional[nn.Module] = nn.ReLU(),
    ) -> None:
        super().__init__()
        dims: List[int] = [input_dim] + list(hidden_dims) + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else nn.Identity()
            layers.append(SpinNeuron(dims[i], dims[i + 1], fib_index=i + 1, activation=act))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

