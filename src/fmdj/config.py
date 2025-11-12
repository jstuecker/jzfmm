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
class FMMConfig():
    # Multipole order:
    p : int = 2

    # important
    max_leaf_size : int = 32
    coarse_fac : float = 4.0
    opening_angle : float = 0.8

    # Memory
    ilist_alloc_fac : int = 512

    # Other
    kahan_summation : bool = False

    # less relevant
    alloc_fac_nodes : float = 1.5
    alloc_min : int = 1024
    stop_coarsen : int = 512

    # untested:
    multipoles_around_com : bool = True

class OldConfig():
    interact_unroll: int | bool = False

    # Memory parameters
    ilist_chunk_fac : float = 4.0  # For ilist evaluation. Will improve this parameter later
    ilist_max_mb : int = 1024      # Maximum memory for ilist evaluation in MB

@dataclass(unsafe_hash=True)
class LoggingConfig():
    level : int = 1
    show_loc : bool = True

@dataclass(unsafe_hash=True)
class Config(VariantConfig):
    # Sub config objects
    fmm : FMMConfig | None = FMMConfig()
    logging : LoggingConfig = LoggingConfig()
    old : OldConfig = OldConfig()

    # flexible objects
    external_potential : PotentialField | None = None

    # parameters
    softening : float = 1e-3

    # Time integration
    centered : int = 100       # If > 0, express positions relative to the #N most bound particles

    def G(self) -> float:
        return 4.30071057317063e-06