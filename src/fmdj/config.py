from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from jztree.config import TreeConfig, LoggingConfig

# ------------------------------------------------------------------------------------------------ #
#                                              Kernels                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class KernelConfig:
    def kind_id(self) -> int:
        raise NotImplementedError
    def params(self, dtype=jnp.float32) -> jax.Array:
        raise NotImplementedError

@dataclass(unsafe_hash=True, slots=True)
class PlummerKernel(KernelConfig):
    """K(r) = -1 / sqrt(r^2 + eps^2)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 0

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class QuarticPlummerKernel(KernelConfig):
    """K(r) = -1 / (r^4 + eps^4)^(1/4)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 1

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class Plummer2DKernel(KernelConfig):
    """K(r) = 0.5 * log(r^2 + eps^2)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 2

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class SoftenedDistanceKernel(KernelConfig):
    """K(r) = sqrt(r^2 + eps^2)."""

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 3

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

# ------------------------------------------------------------------------------------------------ #
#                                              Opening                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class OpeningCriterionConfig:
    def kind_id(self) -> int:
        raise NotImplementedError
    def params(self, dtype=jnp.float32) -> jax.Array:
        raise NotImplementedError

@dataclass(unsafe_hash=True, slots=True)
class OpeningByAngle(OpeningCriterionConfig):
    theta : float = 0.8

    def kind_id(self) -> int:
        return 0

    def params(self, dtype=jnp.float32) -> jax.Array:
        return jnp.asarray([self.theta], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class PotentialField:
    def potential(self, x, t=0., cfg=None):
        """External potential field"""
        raise NotImplementedError
    def acceleration(self, x, t=0., cfg=None):
        """External acceleration field, calculated through autodiff"""
        return -jax.grad(lambda x: jnp.sum(self.potential(x, t=t, cfg=cfg)))(x)

# ------------------------------------------------------------------------------------------------ #
#                                               Units                                              #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class UnitConfig:
    """Simulation units relative to kpc, km/s, and solar masses."""

    pos_in_kpc: float = 1.
    vel_in_kmps: float = 1.
    mass_in_msol: float = 1.

    def G(self) -> float:
        return 4.30071057317063e-06 * self.mass_in_msol / self.pos_in_kpc / self.vel_in_kmps**2

@dataclass(unsafe_hash=True, slots=True)
class IntegratorConfig:
    pass

@dataclass(unsafe_hash=True, slots=True)
class DKDConfig(IntegratorConfig):
    pass

@dataclass(unsafe_hash=True, slots=True)
class KDKConfig(IntegratorConfig):
    pass

@dataclass(unsafe_hash=True, slots=True)
class DKDLatticeConfig(IntegratorConfig):
    dx: float = 1e-4
    dv: float = 1e-4
    int_dtype: type = jnp.int32

# ------------------------------------------------------------------------------------------------ #
#                                               Force                                              #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class DirectSummationConfig:
    kernel : KernelConfig = field(default_factory=PlummerKernel)
    kahan_summation : bool = True

@dataclass(unsafe_hash=True, slots=True)
class FMMConfig:
    # Tree
    tree : TreeConfig = field(
        default_factory=lambda: TreeConfig(
            mass_centered=False, alloc_fac_nodes=1.2, regularization=None, coarse_fac=4.0
        )
    )

    # Kernel
    kernel : KernelConfig = field(default_factory=PlummerKernel)

    # Multipole order:
    p : int = 5

    # Opening criterion
    opening : OpeningCriterionConfig = field(default_factory=OpeningByAngle)

    # Memory
    alloc_fac_ilist : float = 64.
    alloc_fac_comm_nodes : float = 1.5
    alloc_fac_comm_particles : float = 1.5

    # Other
    kahan_summation : bool = False

@dataclass(unsafe_hash=True, slots=True)
class SimConfig:
    # Sub cfg objects
    force : FMMConfig | DirectSummationConfig | None = field(default_factory=FMMConfig)
    units : UnitConfig = field(default_factory=UnitConfig)
    logging : LoggingConfig = field(default_factory=LoggingConfig)

    # flexible objects
    external_potential : PotentialField | None = None

    # Time integration
    integrator: IntegratorConfig = field(default_factory=DKDConfig)
