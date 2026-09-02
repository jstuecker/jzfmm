from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jztree.data import PosMass

def _static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)

@jax.tree_util.register_dataclass
@dataclass
class LocalExpansion:
    """Potential and spatial derivatives evaluated at a set of positions.

    Args:
        values: Array with potential followed by its spatial derivatives along
            the last axis.
        dim: Number of spatial dimensions.
    """

    values: jax.Array

    dim: int = _static_field(default=3)

    def fphi(self) -> jax.Array:
        """Returns force components followed by potential."""
        return jnp.concatenate([self.force(), self.potential()[...,None]], axis=-1)
    def potential(self) -> jax.Array:
        """Returns the potential."""
        return self.values[..., 0]
    def force(self) -> jax.Array:
        """Returns the force, or negative potential gradient."""
        assert self.values.shape[1] >= 1 + self.dim, "Force components not available"
        return -self.values[..., 1:1+self.dim]
    def tide(self) -> jax.Array:
        """Returns the independent components of the negative Hessian."""
        ntide = self.dim * (self.dim + 1) // 2
        assert self.values.shape[1] >= 1 + self.dim + ntide, "Tidal components not available"
        return -self.values[..., 1+self.dim: 1+self.dim+ntide]

@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class Particles:
    """Particle data used by simulations.

    :paramref:`num` and :paramref:`num_total` are only required for multi-GPU
    execution and may remain ``None`` for single-GPU particle data.

    Args:
        pos: Position array of shape ``(size, dim)``.
        mass: Particle masses, either scalar or an array of shape ``(size,)``.
        vel: Velocity array with the same shape as :paramref:`pos`.
        loc: Optional local expansion evaluated at the particle positions.
        num: Number of filled entries on the local device for padded,
            multi-GPU data.
        num_total: Total particle count across all devices for multi-GPU data.
    """

    pos: jax.Array
    mass: jax.Array
    vel : jax.Array

    loc : LocalExpansion | None = None

    num: jax.Array | None = None
    num_total: int | None = _static_field(default=None)
