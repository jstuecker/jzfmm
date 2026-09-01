from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp

from .config import SimConfig, PotentialField

@dataclass(unsafe_hash=True)
class NFWPotential(PotentialField):
    rs: float = 1.0
    rhoc: float = 1.0

    def phic(self, G: float = 1.) -> float:
        return - 4. * jnp.pi * self.rs**2 * self.rhoc * G

    def potential(self, x: jax.Array, t: float = 0., cfg: SimConfig = None) -> jax.Array:
        u = jnp.linalg.norm(x, axis=-1) / self.rs
        # The -1 is to set phi(r->0) = 0. This is numerically beneficial
        return self.phic(G=cfg.units.G()) * (jnp.log(1. + u) / u - 1.)

@dataclass(unsafe_hash=True)
class HernquistPotential(PotentialField):
    a: float = 1.0
    mass: float = 1.0

    def potential(self, x: jax.Array, t: float = 0., cfg: SimConfig = None) -> jax.Array:
        return -cfg.units.G() * self.mass / (jnp.linalg.norm(x, axis=-1) + self.a)

@dataclass(unsafe_hash=True)
class UniformAcceleration(PotentialField):
    acc : tuple[float, float, float] = (0., 0., 0.)

    def potential(self, x: jax.Array, t: float = 0., cfg: SimConfig = None) -> jax.Array:
        return - (self.acc[0] * x[:,0] + self.acc[1] * x[:,1] + self.acc[2] * x[:,2])

@dataclass(unsafe_hash=True)
class DiskPotential(PotentialField):
    """Potential of a disk, approximated by 3 MN potentials as in arxiv:1502.00627"""
    mass: float
    scale_radius: float
    height: float

    def _mn_pars(self):
        coeffs = [{}, {}, {}]
        coeffs[0]["M_over_Md"] = -0.0090, 0.0640, -0.1653, 0.1164, 1.9487
        coeffs[1]["M_over_Md"] =  0.0173,-0.0903,  0.0877, 0.2029,-1.3077
        coeffs[2]["M_over_Md"] = -0.0051, 0.0287, -0.0361,-0.0544, 0.2242
        coeffs[0]["a_over_Rd"] = -0.0358, 0.2610, -0.6987,-0.1193, 2.0074
        coeffs[1]["a_over_Rd"] = -0.0830, 0.4992, -0.7967,-1.2966, 4.4441
        coeffs[2]["a_over_Rd"] = -0.0247, 0.1718, -0.4124,-0.5944, 0.7333

        pars = [{},{},{}]
        for i in range(0,3):
            x = self.height / np.clip(self.scale_radius, 1e-20, None)
            xn = np.array(x)[..., np.newaxis]**np.array([4,3,2,1,0])
            pars[i]["M"] = np.sum(xn*coeffs[i]["M_over_Md"], axis=-1) * self.mass
            pars[i]["a"] = np.sum(xn*coeffs[i]["a_over_Rd"], axis=-1) * self.scale_radius
        
        return pars

    def potential(self, x: jax.Array, t: float = 0., cfg: SimConfig = None) -> jax.Array:
        pars = self._mn_pars()

        R = jnp.sqrt(x[...,0]**2 + x[...,1]**2)
        z = x[...,2]

        pot = 0.
        for p in pars:
            pot = pot - cfg.units.G()*p["M"] / jnp.sqrt(R**2 + (jnp.sqrt(z**2 + self.height**2) + p["a"])**2)

        return pot

@dataclass(unsafe_hash=True)
class MilkyWayPotential(PotentialField):
    halo: PotentialField | None = field(
        default_factory=lambda: NFWPotential(24.17, 4096193.)
    )  # <=> conc=8.71, M200c=1e12
    bulge: PotentialField | None = field(
        default_factory=lambda: HernquistPotential(a=5e2, mass=9e9)
    )
    star_disk: PotentialField | None = field(
        default_factory=lambda: DiskPotential(
            mass=4.1e10, scale_radius=2.5e3, height=3.5e2
        )
    )
    gas_disk: PotentialField | None = field(
        default_factory=lambda: DiskPotential(
            mass=1.9e10, scale_radius=7e3, height=8e1
        )
    )

    def potential(self, x, t=0, cfg=None):
        val = 0.
        if self.halo is not None:
            val = val + self.halo.potential(x, t, cfg)
        if self.bulge is not None:
            val = val + self.bulge.potential(x, t, cfg)
        if self.star_disk is not None:
            val = val + self.star_disk.potential(x, t, cfg)
        if self.gas_disk is not None:
            val = val + self.gas_disk.potential(x, t, cfg)
        return val
