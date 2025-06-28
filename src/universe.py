import math
from dataclasses import dataclass, field
from typing import Generator, List, Tuple

@dataclass
class ExtendedContactFibonacciUniverse:
    """Universe that evolves using Fibonacci-based rotations."""

    position: float = 0.0
    momentum: float = 1.0
    entropy: float = 0.0
    time_phase: float = 0.0
    _fibonacci: List[int] = field(default_factory=lambda: [0, 1])

    def _ensure_fibonacci(self, n: int) -> None:
        """Precompute Fibonacci numbers up to index ``n``."""
        while len(self._fibonacci) <= n:
            self._fibonacci.append(self._fibonacci[-1] + self._fibonacci[-2])

    def step(self, idx: int) -> Tuple[float, float, float, float]:
        """Perform one universe step based on the ``idx``-th Fibonacci number."""
        self._ensure_fibonacci(idx)
        fib = self._fibonacci[idx]
        angle = fib % (2 * math.pi)
        p, m = self.position, self.momentum
        self.position = p * math.cos(angle) - m * math.sin(angle)
        self.momentum = p * math.sin(angle) + m * math.cos(angle)
        self.entropy += math.log(fib + 1)
        self.time_phase += angle
        return self.position, self.momentum, self.entropy, self.time_phase

    def iter_states(self, steps: int) -> Generator[Tuple[float, float, float, float], None, None]:
        """Iterate over the universe for ``steps`` steps, yielding state tuples."""
        for idx in range(1, steps + 1):
            yield self.step(idx)


def main(steps: int = 10) -> List[Tuple[float, float, float, float]]:
    """Run the universe for ``steps`` steps and print the resulting states."""
    universe = ExtendedContactFibonacciUniverse()
    states = []
    for state in universe.iter_states(steps):
        states.append(state)
        print(state)
    return states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the ExtendedContactFibonacciUniverse")
    parser.add_argument("steps", type=int, nargs="?", default=10, help="Number of steps to simulate")
    args = parser.parse_args()
    main(args.steps)
