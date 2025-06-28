import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def fibonacci(n: int) -> int:
    """Compute the n-th Fibonacci number."""
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))  # ~137.5 degrees in radians


def axis_angle_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Create a 3D rotation matrix from an axis and angle.

    Args:
        axis: Tensor of shape (3,)
        angle: rotation angle in radians
    Returns:
        Tensor of shape (3, 3)
    """
    axis = axis / torch.norm(axis)
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c
    return torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s]),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s]),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C]),
    ])


def rotate_vector(x: torch.Tensor, angle: torch.Tensor, axis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rotate vector ``x`` by ``angle`` around ``axis``.

    If ``x`` has dimension 2 or ``axis`` is ``None`` then a 2D rotation on the
    first two components is performed. For dimension >=3 and an ``axis`` the
    first three components are rotated using the axis-angle formulation.
    """
    dim = x.shape[-1]
    if dim < 2:
        return x
    if axis is None or dim == 2:
        c = torch.cos(angle)
        s = torch.sin(angle)
        R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        head = x[..., :2] @ R.T
        tail = x[..., 2:]
        return torch.cat([head, tail], dim=-1)
    else:
        R = axis_angle_rotation_matrix(axis.to(x), angle)
        head = torch.matmul(x[..., :3], R.T)
        tail = x[..., 3:]
        return torch.cat([head, tail], dim=-1)


class SpinNeuron(nn.Module):
    """A neuron that performs a Fibonacci-scaled golden-angle rotation before
    a linear transformation and activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fib_index: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        axis: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation if activation is not None else nn.Identity()
        self.fib_index = fib_index
        if axis is not None:
            axis = axis.to(dtype=torch.float)
        self.register_buffer("axis", axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle = GOLDEN_ANGLE * fibonacci(self.fib_index)
        axis = self.axis if self.axis is not None else None
        rotated = rotate_vector(x, angle, axis)
        out = self.linear(rotated)
        out = self.activation(out)
        return out

