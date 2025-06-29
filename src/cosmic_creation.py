"""Cosmic creation algorithm inspired by extended-contact physics.

This module implements a simple simulation based on the
"Brahman" and "UniverseDS" idea described in the repository's
conversation. The algorithm evolves a universe using
Fibonacci-scaled rotations, entropy growth and a cyclic time phase.
Each step is written to an abstract key-value store (Brahman) and
added as a node in the Universe data structure.

This is a conceptual demonstration and not intended for real
physical modelling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


GOLDEN_RATIO = (1 + 5 ** 0.5) / 2
GOLDEN_ANGLE = 2 * math.pi * (1 - 1 / GOLDEN_RATIO)


class Brahman:
    """Minimal key-value store representing raw cosmic memory."""

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, float]] = {}

    def write(self, key: str, value: Dict[str, float]) -> None:
        self.store[key] = value

    def read(self, key: str) -> Dict[str, float]:
        return self.store.get(key, {})


@dataclass
class Node:
    q: float
    p: float
    entropy: float
    phase: float
    soul: float = 0.0


@dataclass
class UniverseDS:
    """Simple universe data structure built during the simulation."""

    points: List[Node] = field(default_factory=list)
    connectivity: List[Tuple[int, int]] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        if self.points:
            self.connectivity.append((len(self.points) - 1, len(self.points)))
        self.points.append(node)


class CosmicCreation:
    """Simulate the creation of a universe writing to Brahman."""

    def __init__(
        self,
        steps: int = 10,
        dt: float = 0.1,
        entropy_rate: float = 0.01,
        init_q: float = 1.0,
        init_p: float = 0.0,
        init_entropy: float = 0.0,
        init_phase: float = 0.0,
        init_soul: float = 0.0,
    ) -> None:
        self.steps = steps
        self.dt = dt
        self.entropy_rate = entropy_rate
        self.q = init_q
        self.p = init_p
        self.entropy = init_entropy
        self.phase = init_phase
        self.soul = init_soul

        self.brahman = Brahman()
        self.universe = UniverseDS()
        self._fibs: List[int] = [0, 1]

    def _ensure_fibonacci(self, n: int) -> None:
        while len(self._fibs) <= n:
            self._fibs.append(self._fibs[-1] + self._fibs[-2])

    def _rotation_angle(self, idx: int) -> float:
        self._ensure_fibonacci(idx)
        return (self._fibs[idx] * GOLDEN_ANGLE) % (2 * math.pi)

    def step(self, idx: int) -> Node:
        angle = self._rotation_angle(idx)
        c = math.cos(angle)
        s = math.sin(angle)
        q, p = self.q, self.p
        self.q = q * c - p * s
        self.p = q * s + p * c
        self.phase = (self.phase + self.dt) % (2 * math.pi)
        self.entropy += self.entropy_rate
        node = Node(self.q, self.p, self.entropy, self.phase, self.soul)
        self.brahman.write(
            f"step_{idx}",
            {"q": self.q, "p": self.p, "S": self.entropy, "phi": self.phase, "sigma": self.soul},
        )
        self.universe.add_node(node)
        return node

    def run(self) -> UniverseDS:
        for idx in range(self.steps):
            self.step(idx)
        return self.universe


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the CosmicCreation simulation")
    parser.add_argument("steps", type=int, nargs="?", default=10, help="Number of steps")
    args = parser.parse_args()

    sim = CosmicCreation(steps=args.steps)
    universe = sim.run()
    for i, node in enumerate(universe.points):
        print(f"Step {i}: q={node.q:.3f} p={node.p:.3f} S={node.entropy:.3f} phi={node.phase:.3f}")


if __name__ == "__main__":
    main()
