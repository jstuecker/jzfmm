from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jztree.data import PosMass

def _static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)

@jax.tree_util.register_dataclass
@dataclass
class LocalExpansion:
    values: jax.Array

    dim: int = _static_field(default=3)

    def fphi(self):
        return jnp.concatenate([self.force(), self.potential()[...,None]], axis=-1)
    def potential(self):
        return self.values[..., 0]
    def force(self):
        assert self.values.shape[1] >= 1 + self.dim, "Force components not available"
        return -self.values[..., 1:1+self.dim]
    def tide(self):
        ntide = self.dim * (self.dim + 1) // 2
        assert self.values.shape[1] >= 1 + self.dim + ntide, "Tidal components not available"
        return -self.values[..., 1+self.dim: 1+self.dim+ntide]

@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class Particles:
    pos: jax.Array
    mass: jax.Array
    vel : jax.Array

    loc : LocalExpansion | None = None

    num: jax.Array | None = None
    num_total: int | None = _static_field(default=None)
