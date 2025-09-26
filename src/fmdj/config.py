from dataclasses import dataclass
from .variants import VariantConfig

@dataclass(unsafe_hash=True)
class OpeningBarnesAndHut:
    opening_angle : float = 0.8

@dataclass(unsafe_hash=True)
class OpeningRelative:
    relative_accuracy : float = 0.01

@dataclass(unsafe_hash=True)
class Config(VariantConfig):
    # Tree construction
    p : int = 2                # Multipole order
    max_leaf_size : int = 64   # Leaves will have at most this many particles
    softening : float = 1e-3   # Plummer softening length

    # Tree Walk
    opening: OpeningBarnesAndHut | OpeningRelative = OpeningBarnesAndHut()

    # Memory parameters
    ilist_chunk_fac : float = 4.0  # For ilist evaluation. Will improve this parameter later
    ilist_max_mb : int = 1024      # Maximum memory for ilist evaluation in MB

    # Optional outputs
    get_potential : bool = False