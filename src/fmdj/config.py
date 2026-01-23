from dataclasses import dataclass
import jax
import jax.numpy as jnp

from jztree.config import TreeConfig, LoggingConfig

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
    p : int = 3

    # important
    opening_angle : float = 0.85

    # Memory
    ilist_alloc_fac : int = 512

    # Other
    kahan_summation : bool = False


class OldConfig():
    interact_unroll: int | bool = False

    # Memory parameters
    ilist_chunk_fac : float = 4.0  # For ilist evaluation. Will improve this parameter later
    ilist_max_mb : int = 1024      # Maximum memory for ilist evaluation in MB

@dataclass(unsafe_hash=True)
class Config():
    # Sub config objects
    tree : TreeConfig | None = TreeConfig()
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