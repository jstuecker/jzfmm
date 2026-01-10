from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from .config import Config, PotentialField

@dataclass(unsafe_hash=True)
class NFWPotential(PotentialField):
    rs: float = 1.0
    rhoc: float = 1.0

    def phic(self, G: float = 1.) -> float:
        return - 4. * jnp.pi * self.rs**2 * self.rhoc * G

    def potential(self, x: jax.Array, t: float = 0., cfg: Config = None) -> jax.Array:
        u = jnp.linalg.norm(x, axis=-1) / self.rs
        # The -1 is to set phi(r->0) = 0. This is numerically beneficial
        return self.phic(G=cfg.G()) * (jnp.log(1. + u) / u - 1.)
    
@dataclass(unsafe_hash=True)
class HernquistPotential(PotentialField):
    a: float = 1.0
    M: float = 1.0

    def potential(self, x: jax.Array, t: float = 0., cfg: Config = None) -> jax.Array:
        return -cfg.G() * self.M / (jnp.linalg.norm(x, axis=-1) + self.a)
    
@dataclass(unsafe_hash=True)
class UniformAcceleration(PotentialField):
    acc : tuple[float, float, float] = (0., 0., 0.)

    def potential(self, x: jax.Array, t: float = 0., cfg: Config = None) -> jax.Array:
        return - (self.acc[0] * x[:,0] + self.acc[1] * x[:,1] + self.acc[2] * x[:,2])