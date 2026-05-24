from dataclasses import dataclass, field
import math
import jax
import jax.numpy as jnp

from jztree.config import TreeConfig, LoggingConfig

# ------------------------------------------------------------------------------------------------ #
#                                              Kernels                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True)
class KernelConfig:
    def kind_id(self) -> int:
        raise NotImplementedError
    def params(self, dtype=jnp.float32) -> jax.Array:
        raise NotImplementedError
    def self_value(self) -> float:
        raise NotImplementedError

@dataclass(unsafe_hash=True)
class PlummerKernel(KernelConfig):
    """K(r) = -1 / sqrt(r^2 + eps^2)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 0

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

    def self_value(self) -> float:
        return -1.0 / self.softening

@dataclass(unsafe_hash=True)
class QuarticPlummerKernel(KernelConfig):
    """K(r) = -1 / (r^4 + eps^4)^(1/4)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 1

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

    def self_value(self) -> float:
        return -1.0 / self.softening

@dataclass(unsafe_hash=True)
class Plummer2DKernel(KernelConfig):
    """K(r) = 0.5 * log(r^2 + eps^2)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 2

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

    def self_value(self) -> float:
        return math.log(self.softening)

@dataclass(unsafe_hash=True)
class SoftenedDistanceKernel(KernelConfig):
    """K(r) = sqrt(r^2 + eps^2)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 3

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

    def self_value(self) -> float:
        return self.softening

# ------------------------------------------------------------------------------------------------ #
#                                              Opening                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True)
class OpeningCriterionConfig:
    def kind_id(self) -> int:
        raise NotImplementedError
    def params(self, dtype=jnp.float32) -> jax.Array:
        raise NotImplementedError

@dataclass(unsafe_hash=True)
class OpeningByAngle(OpeningCriterionConfig):
    theta : float = 0.8

    def kind_id(self) -> int:
        return 0

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.theta], dtype=dtype)

@dataclass(unsafe_hash=True)
class PotentialField:
    def potential(self, x, t=0., cfg=None):
        """External potential field"""
        raise NotImplementedError
    def acceleration(self, x, t=0., cfg=None):
        """External acceleration field, calculated through autodiff"""
        return -jax.grad(lambda x: jnp.sum(self.potential(x, t=t, cfg=cfg)))(x)

# ------------------------------------------------------------------------------------------------ #
#                                                FMM                                               #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True)
class FMMConfig():
    # Multipole order:
    p : int = 4
    p_extra_m2l : int = 0

    # Opening criterion
    opening : OpeningCriterionConfig = field(default_factory=OpeningByAngle)

    # Memory
    alloc_fac_ilist : float = 256.

    # Other
    kahan_summation : bool = False


@dataclass(unsafe_hash=True)
class Config():
    # Sub config objects
    tree : TreeConfig | None = TreeConfig(mass_centered=False)
    fmm : FMMConfig | None = FMMConfig()
    logging : LoggingConfig = LoggingConfig()

    # flexible objects
    external_potential : PotentialField | None = None
    kernel : KernelConfig = field(default_factory=PlummerKernel)

    # Time integration
    centered : int = 100       # If > 0, express positions relative to the #N most bound particles

    def G(self) -> float:
        return 4.30071057317063e-06
