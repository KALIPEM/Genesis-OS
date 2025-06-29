"""Example script demonstrating the CosmicCreation simulation."""

import sys
from pathlib import Path

# Ensure the src package is importable when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cosmic_creation import CosmicCreation


def main() -> None:
    sim = CosmicCreation(steps=5)
    universe = sim.run()
    for i, node in enumerate(universe.points):
        print(
            f"Step {i}: q={node.q:.3f} p={node.p:.3f} S={node.entropy:.3f} phi={node.phase:.3f}"
        )


if __name__ == "__main__":
    main()
