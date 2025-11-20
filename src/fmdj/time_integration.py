import jax.numpy as jnp
import jax
import fmdj
from fmdj import Config
from dataclasses import dataclass, replace
from typing import Generator
import time
from functools import partial

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
        return self.pos if self.cpos is None else self.pos + self.cpos
    def avel(self):
        return self.vel if self.cvel is None else self.vel + self.cvel

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
find_center.jit = jax.jit(find_center, static_argnames=("cfg",))

def shift_reference_center(p : Particles, cfg : Config = None):
    dcpos, dcvel = find_center(p, cfg=cfg)

    cpos = p.cpos + dcpos if p.cpos is not None else dcpos
    cvel = p.cvel + dcvel if p.cvel is not None else dcvel
    
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
        p.acc, p.pot = fmdj.fmm.get_force_and_potential.jit(p.pos, p.mass, cfg=cfg, separately=True)

    vh = kick(p.vel, p.acc + ext_acc(p, t, cfg), 0.5*dt, mask=mask)
    p.pos = drift(p.pos, vh, dt, mask=mask)
    if p.cpos is not None:
        p.cpos = p.cpos + p.cvel * dt

    p.acc, p.pot = fmdj.fmm.get_force_and_potential.jit(p.pos, p.mass, cfg=cfg, separately=True)
    p.vel = kick(vh, p.acc + ext_acc(p, t + dt, cfg), 0.5*dt, mask=mask)

    if cfg.centered:
        p = shift_reference_center(p, cfg=cfg)

    return p
timestep.jit = jax.jit(timestep, static_argnames=("cfg",))

def simulate(p: Particles, tend: float, nsteps: int, cfg: Config, tstart: float = 0.) -> Particles:
    # Make an initial dt=0 step to get the correct initial acceleration
    p = fmdj.time_integration.timestep(p, dt=0., cfg=cfg, t=tstart)
    dt = (tend - tstart) / nsteps
    def step(i, carry):
        p, t = carry
        return fmdj.time_integration.timestep.jit(p, dt=dt, cfg=cfg, t=t), t + dt

    p, t = jax.lax.fori_loop(0, nsteps, step, (p, tstart))
    return p
simulate.jit = jax.jit(simulate, static_argnames=("cfg",))
simulate.vjp = jax.custom_vjp(simulate, nondiff_argnames=("tend", "nsteps", "cfg", "tstart"))

def simulate_fwd(p: Particles, tend: float, nsteps: int, cfg: Config, tstart: float = 0.) -> Particles:
    pcom = jnp.mean(p.apos(), axis=0)
    fmdj.log("Starting forward pass, <pos> = ({:.2f},{:.2f},{:.2f})",
             pcom[0], pcom[1], pcom[2], level=1, cfg=cfg)

    p = simulate(p, tend, nsteps, cfg, tstart)

    pcom = jnp.mean(p.apos(), axis=0)
    fmdj.log("Finished forward pass, <pos> = ({:.2f},{:.2f},{:.2f})",
             pcom[0], pcom[1], pcom[2], level=1, cfg=cfg)

    return p, p

def simulate_bwd(tend: float, nsteps: int, cfg: Config, tstart: float, p: Particles, gp: jnp.ndarray):
    dt = (tend - tstart) / nsteps

    def step(i, carry):
        p, t, gp = carry

        p = fmdj.time_integration.timestep.jit(p, dt=-dt, cfg=cfg, t=t)
        _, vjp_fun = jax.vjp(lambda p: fmdj.time_integration.timestep.jit(p, dt=dt, cfg=cfg, t=t), p)
        gxp, = vjp_fun(gp)
        return p, t + dt, gxp

    p_prev, t_prev, gp_prev = jax.lax.fori_loop(0, nsteps, step, (p, tend, gp))

    pcom = jnp.mean(p_prev.apos(), axis=0)
    fmdj.log("Finished backward pass, <pos> = ({:.2f},{:.2f},{:.2f})",
             pcom[0], pcom[1], pcom[2], level=1, cfg=cfg)
    
    return (gp_prev,)

simulate.vjp.defvjp(simulate_fwd, simulate_bwd)

def clean_particles(p: Particles) -> Particles:
    p = replace(p)  # Make a copy to avoid modifying the input

    p.pos = jnp.asarray(p.pos)
    p.vel = jnp.asarray(p.vel)
    p.mass = jnp.asarray(p.mass)

    if p.pot is None:
        p.pot = jnp.zeros_like(p.mass)
    if p.acc is None:
        p.acc = jnp.zeros_like(p.pos)
    return p

def simulate_with_outputs(
        p : Particles, 
        tend: float, 
        nout: int, 
        steps_per_output: int,
        cfg: Config,
        tstart: float = 0.
    ) -> Generator[Particles, None, None]:
    """Don't jit this function!"""
    p = clean_particles(p) # This helps avoiding double jit-compilations

    tp0 = time.perf_counter()
    fmdj.log("Compiling jitted simulation...", level=1, cfg=cfg)
    simulate.jit.lower(p, tend=jnp.float32(0.1), nsteps=steps_per_output, cfg=cfg, tstart=jnp.float32(0.1)).compile()
    fmdj.log("Compilation done after {:.2f}s", time.perf_counter()-tp0, level=1, cfg=cfg)

    yield tstart, p

    for isnap in range(nout):
        t0 = jnp.float32(tstart + isnap * (tend/nout))
        t1 = jnp.float32(tstart + (isnap+1) * (tend/nout))
        tpa = time.perf_counter()
        p = simulate.jit(p, tend=t1, nsteps=steps_per_output, cfg=cfg, tstart=t0)
        fmdj.log("Reached output {} ({:.2f}s for {} steps)", 
                 isnap+1, time.perf_counter()-tpa, steps_per_output, level=1, cfg=cfg)
        yield t1, p
    
    fmdj.log("Total simulation time: {:.2f}s for {} steps", 
             time.perf_counter()-tp0, nout*steps_per_output, level=1, cfg=cfg)