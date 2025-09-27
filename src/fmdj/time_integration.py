import jax.numpy as jnp
import jax
import fmdj
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass
class Particles():
    pos : jnp.ndarray
    vel : jnp.ndarray
    mass : jnp.ndarray
    acc : jnp.ndarray | None = None
    pot : jnp.ndarray | None = None

def kick(vel, acc, dt, mask=None):
    if mask is not None:
        return vel + jnp.where(mask[:,None], acc, 0) * dt
    else:
        return vel + acc * dt

def drift(pos, vel, dt, mask=None):
    if mask is not None:
        return pos + jnp.where(mask[:,None], vel, 0) * dt
    else:
        return pos + vel * dt

def timestep(p : Particles, dt, cfg : fmdj.config.Config, mask=None):
    if p.acc is None:
        p.acc, p.pot = fmdj.multipoles.direct_summation_force.jit(p.pos, p.mass, cfg=cfg)

    vh = kick(p.vel, p.acc, 0.5*dt, mask=mask)
    p.pos = drift(p.pos, vh, dt, mask=mask)

    p.acc, p.pot = fmdj.multipoles.direct_summation_force.jit(p.pos, p.mass, cfg=cfg)
    p.vel = kick(vh, p.acc, 0.5*dt, mask=mask)

    return p
timestep.jit = jax.jit(timestep, static_argnames=("cfg",))

def simulate(p : Particles, tmax, nsteps, cfg):
    # Make an initial dt=0 step to get the correct initial acceleration
    p = fmdj.time_integration.timestep(p, dt=0., cfg=cfg)
    def step(i, p : Particles):
        return fmdj.time_integration.timestep.jit(p, dt=tmax/nsteps, cfg=cfg)

    p = jax.lax.fori_loop(0, nsteps, step, p)
    return p
simulate.jit = jax.jit(simulate, static_argnames=("nsteps","cfg"))