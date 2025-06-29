"""Genesis-OS core modules."""

from .cosmic_creation import CosmicCreation, UniverseDS, Brahman
from .universe import ExtendedContactFibonacciUniverse

__all__ = [
    "CosmicCreation",
    "UniverseDS",
    "Brahman",
    "ExtendedContactFibonacciUniverse",
]

try:
    from .spin_network import SpinNetwork
    from .spin_neuron import SpinNeuron
    __all__.extend(["SpinNetwork", "SpinNeuron"])
except Exception:  # torch may not be installed
    SpinNetwork = None
    SpinNeuron = None
