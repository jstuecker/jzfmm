from .config import Config, PotentialField
import jax.numpy as jnp
import jax
from dataclasses import dataclass

@dataclass(unsafe_hash=True)
class NFWPotential(PotentialField):
    rs : float = 1.0
    rhoc : float = 1.0

    def phic(self, G=1.):
        return - 4. * jnp.pi * self.rs**2 * self.rhoc * G

    def potential(self, x, t=0., cfg=None):
        u = jnp.linalg.norm(x, axis=-1) / self.rs
        # The -1 is to set phi(r->0) = 0. This is numerically beneficial
        return self.phic(G=cfg.G()) * (jnp.log(1. + u) / u - 1.)