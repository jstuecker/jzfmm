from __future__ import annotations

from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from jztree.config import TreeConfig, LoggingConfig

# ------------------------------------------------------------------------------------------------ #
#                                              Kernels                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class KernelConfig:
    """Base class for radial interaction-kernel configurations."""

    def kind_id(self) -> int:
        """Returns the backend identifier of the kernel."""
        raise NotImplementedError
    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        """Returns kernel parameters in the requested dtype."""
        raise NotImplementedError

@dataclass(unsafe_hash=True, slots=True)
class PlummerKernel(KernelConfig):
    r"""Plummer-softened inverse-distance kernel.

    Implements :math:`K(r)=-1/\sqrt{r^2+\epsilon^2}`.

    Args:
        softening: Softening length :math:`\epsilon`.
    """

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 0

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class Plummer2DKernel(KernelConfig):
    r"""Two-dimensional Plummer-softened logarithmic kernel.

    Implements :math:`K(r)=\frac{1}{2}\log(r^2+\epsilon^2)`.

    Args:
        softening: Softening length :math:`\epsilon`.
    """

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 1

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class SoftenedDistanceKernel(KernelConfig):
    r"""Softened distance kernel.

    Implements :math:`K(r)=\sqrt{r^2+\epsilon^2}`.

    Args:
        softening: Softening length :math:`\epsilon`.
    """

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 2

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

# ------------------------------------------------------------------------------------------------ #
#                                              Opening                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class OpeningCriterionConfig:
    """Base class for FMM opening-criterion configurations."""

    def kind_id(self) -> int:
        """Returns the backend identifier of the opening criterion."""
        raise NotImplementedError
    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        """Returns criterion parameters in the requested dtype."""
        raise NotImplementedError

@dataclass(unsafe_hash=True, slots=True)
class OpeningByAngle(OpeningCriterionConfig):
    """Geometric opening criterion based on an opening angle.

    Args:
        theta: Maximum opening angle. Smaller values increase accuracy and
            computational cost.
    """

    theta : float = 0.8

    def kind_id(self) -> int:
        return 0

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.theta], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class PotentialField:
    """Base class for external potential fields.

    Assign an instance to :paramref:`SimConfig.external_potential` to add its
    acceleration during time integration. Subclasses normally implement
    :meth:`potential`; :meth:`acceleration` obtains its negative gradient with
    autodiff. Built-in fields are provided by :mod:`jzfmm.external_potential`.

    The external contribution is applied during integration and is not included
    in the :class:`jzfmm.data.LocalExpansion` returned for particle
    self-interactions.
    """

    def potential(
        self, x: jax.Array, t: float | jax.Array = 0., cfg: SimConfig | None = None
    ) -> jax.Array:
        """Evaluates the potential at positions :paramref:`x`."""
        raise NotImplementedError
    def acceleration(
        self, x: jax.Array, t: float | jax.Array = 0., cfg: SimConfig | None = None
    ) -> jax.Array:
        """Evaluates acceleration as the negative potential gradient."""
        return -jax.grad(lambda x: jnp.sum(self.potential(x, t=t, cfg=cfg)))(x)

# ------------------------------------------------------------------------------------------------ #
#                                               Units                                              #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class UnitConfig:
    """Defines simulation units relative to common astrophysical units.

    Args:
        pos_in_kpc: Length represented by one simulation position unit in kpc.
        vel_in_kmps: Speed represented by one simulation velocity unit in km/s.
        mass_in_msol: Mass represented by one simulation mass unit in solar
            masses.
    """

    pos_in_kpc: float = 1.
    vel_in_kmps: float = 1.
    mass_in_msol: float = 1.

    def G(self) -> float:
        """Returns the gravitational constant in simulation units."""
        return 4.30071057317063e-06 * self.mass_in_msol / self.pos_in_kpc / self.vel_in_kmps**2

@dataclass(unsafe_hash=True, slots=True)
class IntegratorConfig:
    """Base class for time-integrator configurations."""

@dataclass(unsafe_hash=True, slots=True)
class DKDConfig(IntegratorConfig):
    """Drift-kick-drift leapfrog integrator configuration."""

@dataclass(unsafe_hash=True, slots=True)
class KDKConfig(IntegratorConfig):
    """Kick-drift-kick leapfrog integrator configuration."""

@dataclass(unsafe_hash=True, slots=True)
class DKDLatticeConfig(IntegratorConfig):
    """Drift-kick-drift integrator using integer phase-space coordinates.

    Args:
        dx: Position lattice spacing.
        dv: Velocity lattice spacing.
        int_dtype: Integer dtype used for lattice coordinates. Both
            :class:`jax.numpy.int32` and :class:`jax.numpy.int64` are supported,
            independently of the floating-point format used elsewhere.
    """

    dx: float = 1e-4
    dv: float = 1e-4
    int_dtype: type = jnp.int32

# ------------------------------------------------------------------------------------------------ #
#                                               Force                                              #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class DirectSummationConfig:
    """Configures direct pair summation.

    Args:
        kernel: Radial interaction kernel.
        kahan_summation: Whether to use compensated summation.
        remove_self_interaction: Whether to exclude each particle's interaction
            with itself.
    """

    kernel : KernelConfig = field(default_factory=PlummerKernel)
    kahan_summation : bool = True
    remove_self_interaction : bool = True

@dataclass(unsafe_hash=True, slots=True)
class FMMConfig:
    """Configures fast-multipole force evaluation.

    Args:
        tree: Tree construction configuration.
        kernel: Radial interaction kernel.
        p: Multipole expansion order.
        opening: Node-opening criterion.
        alloc_fac_ilist: Interaction-list entries allocated per leaf-node
            buffer entry; the total capacity is approximately this factor
            times the allocated number of leaf nodes.
        alloc_fac_comm_nodes: Node communication-buffer capacity as a multiple
            of the local node-buffer size.
        alloc_fac_comm_particles: Particle communication-buffer capacity as a
            multiple of the local particle-buffer size.
        kahan_summation: Whether to use compensated summation where available.
        remove_self_interaction: Whether to exclude each particle's interaction
            with itself.
    """

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
    remove_self_interaction : bool = True

@dataclass(unsafe_hash=True, slots=True)
class SimConfig:
    """Collects force, unit, logging, and integration configuration.

    Args:
        force: Force configuration, or ``None`` to disable self-gravity.
        units: Simulation unit configuration.
        logging: Logging configuration.
        external_potential: Optional external potential field.
        integrator: Time-integrator configuration.
    """

    # Sub cfg objects
    force : FMMConfig | DirectSummationConfig | None = field(default_factory=FMMConfig)
    units : UnitConfig = field(default_factory=UnitConfig)
    logging : LoggingConfig = field(default_factory=LoggingConfig)

    # flexible objects
    external_potential : PotentialField | None = None

    # Time integration
    integrator: IntegratorConfig = field(default_factory=DKDConfig)
