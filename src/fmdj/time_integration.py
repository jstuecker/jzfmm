import jax.numpy as jnp
import jax
import fmdj
from fmdj import Config
from dataclasses import dataclass, replace

@jax.tree_util.register_dataclass
@dataclass
class Particles():
    pos : jnp.ndarray
    vel : jnp.ndarray
    mass : jnp.ndarray
    acc : jnp.ndarray | None = None
    pot : jnp.ndarray | None = None

    cpos : jnp.ndarray | None = None
    cvel : jnp.ndarray | None = None

    def apos(self):
        return self.pos if self.cpos is None else self.pos - self.cpos
    def avel(self):
        return self.vel if self.cvel is None else self.vel - self.cvel

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

def find_center(p : Particles, cfg : Config = None):
    ebind = p.pot + 0.5 * jnp.sum(p.vel**2, axis=-1)
    if cfg.centered > 1:
        # sort by most boundedness
        i = jnp.argsort(ebind)
        # average over the N=cfg.centered most bound particles
        cpos = jnp.mean(p.pos[i[:cfg.centered]], axis=0)
        cvel = jnp.mean(p.vel[i[:cfg.centered]], axis=0)
    elif cfg.centered == 1:
        i = jnp.argmin(ebind)
        cpos = p.pos[i]
        cvel = p.vel[i]
    else:
        cpos = jnp.zeros(3)
        cvel = jnp.zeros(3)

    return cpos, cvel

def shift_reference_center(p : Particles, cfg : Config = None):
    cpos, cvel = find_center(p, cfg=cfg)

    dcpos = cpos - p.cpos if p.cpos is not None else cpos
    dcvel = cvel - p.cvel if p.cvel is not None else cvel
    
    # Shift to the new frame
    p = replace(p, pos=p.pos-dcpos, vel=p.vel-dcvel, cpos=cpos, cvel=cvel)

    return p

def ext_acc(p : Particles, t, cfg : Config):
    if cfg.external_potential is None:
        return jnp.zeros_like(p.pos)
    else:
        acc = cfg.external_potential.acceleration(p.apos(), t=t, cfg=cfg)

        ## Later also think about centering here
        
        return acc

def timestep(p : Particles, dt, cfg : Config, t=0., mask=None):
    p = replace(p)  # Make a copy to avoid modifying the input

    if p.acc is None:
        p.acc, p.pot = fmdj.multipoles.direct_summation_force.jit(p.pos, p.mass, cfg=cfg)

    vh = kick(p.vel, p.acc + ext_acc(p, t, cfg), 0.5*dt, mask=mask)
    p.pos = drift(p.pos, vh, dt, mask=mask)

    p.acc, p.pot = fmdj.multipoles.direct_summation_force.jit(p.pos, p.mass, cfg=cfg)
    p.vel = kick(vh, p.acc + ext_acc(p, t + dt, cfg), 0.5*dt, mask=mask)

    if cfg.centered:
        p = shift_reference_center(p, cfg=cfg)

    return p
timestep.jit = jax.jit(timestep, static_argnames=("cfg",))

def simulate(p : Particles, tmax, nsteps, cfg, tstart=0.):
    # Make an initial dt=0 step to get the correct initial acceleration
    p = fmdj.time_integration.timestep(p, dt=0., cfg=cfg, t=tstart)
    def step(i, carry):
        p, t = carry
        return fmdj.time_integration.timestep.jit(p, dt=tmax/nsteps, cfg=cfg, t=t), t + tmax/nsteps

    p, t = jax.lax.fori_loop(0, nsteps, step, (p, tstart))
    return p
simulate.jit = jax.jit(simulate, static_argnames=("cfg",))