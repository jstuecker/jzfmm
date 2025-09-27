import jax.numpy as jnp
import jax
import fmdj
from dataclasses import dataclass

@jax.tree_util.register_dataclass
@dataclass
class Particles():
    pos : jnp.ndarray
    vel : jnp.ndarray
    m : jnp.ndarray
    acc : jnp.ndarray | None = None
    phi : jnp.ndarray | None = None

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

def timestep(pos, vel, m, dt, cfg : fmdj.config.Config, mask=None, acc=None):
    if acc is None:
        acc, phi = fmdj.multipoles.direct_summation_force.jit(pos, m, cfg=cfg)

    vh = kick(vel, acc, 0.5*dt, mask=mask)
    pos = drift(pos, vh, dt, mask=mask)

    acc, phi = fmdj.multipoles.direct_summation_force.jit(pos, m, cfg=cfg)
    vel = kick(vh, acc, 0.5*dt, mask=mask)

    return pos, vel, acc
timestep.jit = jax.jit(timestep, static_argnames=("cfg",))

def simulate(pos0, vel0, m, tmax, nsteps, cfg):
    # Make an initial dt=0 step to get the correct initial acceleration
    pos, vel, acc = fmdj.time_integration.timestep(pos0, vel0, m, dt=0., cfg=cfg, acc=None)
    def step(i, carry):
        pos, vel, acc = carry
        return fmdj.time_integration.timestep.jit(pos, vel, m, dt=tmax/nsteps, cfg=cfg, acc=acc)
    pos, vel, acc = jax.lax.fori_loop(0, nsteps, step, (pos, vel, acc))
    return pos, vel
simulate.jit = jax.jit(simulate, static_argnames=("nsteps","cfg"))