import jax
import jax.numpy as jnp
from dataclasses import dataclass
from .variants import VariantConfig

@dataclass(unsafe_hash=True)
class OpeningBarnesAndHut:
    opening_angle : float = 0.8

@dataclass(unsafe_hash=True)
class OpeningRelative:
    relative_accuracy : float = 0.01

@dataclass(unsafe_hash=True)
class PotentialField:
    def potential(self, x, t=0., cfg=None):
        """External potential field"""
        raise NotImplementedError
    def acceleration(self, x, t=0., cfg=None):
        """External acceleration field, calculated through autodiff"""
        return -jax.grad(lambda x: jnp.sum(self.potential(x, t=t, cfg=cfg)))(x)

@dataclass(unsafe_hash=True)
class TreeConfig():
    # Multipole order:
    p : int = 2

    # important
    max_leaf_size : int = 64
    coarse_fac : float = 4.0

    ilist_alloc_fac : int = 512

    # less relevant
    alloc_fac_nodes : float = 1.0
    alloc_min : int = 128
    stop_coarsen : int = 512

    # untested:
    multipoles_around_com : bool = True

@dataclass(unsafe_hash=True)
class LoggingConfig():
    level : int = 1
    show_loc : bool = True

@dataclass(unsafe_hash=True)
class Config(VariantConfig):
    # New tree parameters
    tree : TreeConfig = TreeConfig()
    logging : LoggingConfig = LoggingConfig()

    # everything below might be outdated

    # Physical parameters
    external_potential : PotentialField | None = None

    # Tree construction
    p : int = 2                # Multipole order
    max_leaf_size : int = 64   # Leaves will have at most this many particles
    softening : float = 1e-3   # Plummer softening length

    # Tree Walk
    opening: OpeningBarnesAndHut | OpeningRelative = OpeningBarnesAndHut()

    # Time integration
    centered : int = 100       # If > 0, express positions relative to the #N most bound particles

    # Memory parameters
    ilist_chunk_fac : float = 4.0  # For ilist evaluation. Will improve this parameter later
    ilist_max_mb : int = 1024      # Maximum memory for ilist evaluation in MB

    # Optional outputs
    get_potential : bool = True

    def G(self) -> float:
        return 4.30071057317063e-06