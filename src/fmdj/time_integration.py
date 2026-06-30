from dataclasses import replace
from typing import Generator
import time
import warnings
import jax.numpy as jnp
import jax

from jztree.tools import log
from .data import Particles, LocalExpansion
from .config import DirectSummationConfig, FMMConfig, SimConfig
from .fmm import direct_summation, fast_multipole_method

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

def force_and_potential(p: Particles, cfg: SimConfig) -> LocalExpansion:
    cfg_force = cfg.force
    if cfg_force is None:
        dim = p.pos.shape[-1]
        values = jnp.zeros(p.pos.shape[:-1] + (dim + 1,), dtype=p.pos.dtype)
        return LocalExpansion(values, dim=dim)
    if isinstance(cfg_force, DirectSummationConfig):
        return direct_summation(p, cfg_direct=cfg_force, G=cfg.units.G())
    if isinstance(cfg_force, FMMConfig):
        return fast_multipole_method(p, cfg_fmm=cfg_force, G=cfg.units.G(), pout=1)
    raise TypeError(f"Unsupported force cfg type {type(cfg_force)}")
force_and_potential.jit = jax.jit(force_and_potential, static_argnames=("cfg",))

def find_center(p : Particles, cfg: SimConfig = None):
    ebind = p.loc.potential() + 0.5 * jnp.sum(p.vel**2, axis=-1)
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
        cpos = jnp.zeros(p.pos.shape[-1], dtype=p.pos.dtype)
        cvel = jnp.zeros(p.vel.shape[-1], dtype=p.vel.dtype)

    return cpos, cvel
find_center.jit = jax.jit(find_center, static_argnames=("cfg",))

def shift_reference_center(p : Particles, cfg: SimConfig = None):
    dcpos, dcvel = find_center(p, cfg=cfg)

    cpos = p.cpos + dcpos if p.cpos is not None else dcpos
    cvel = p.cvel + dcvel if p.cvel is not None else dcvel
    
    # Shift to the new frame
    p = replace(p, pos=p.pos-dcpos, vel=p.vel-dcvel, cpos=cpos, cvel=cvel)

    return p

def ext_acc(p : Particles, t, cfg: SimConfig):
    if cfg.external_potential is None:
        return jnp.zeros_like(p.pos)
    else:
        acc = cfg.external_potential.acceleration(p.apos(), t=t, cfg=cfg)

        ## Later also think about centering here
        
        return acc

def timestep(p : Particles, dt, cfg: SimConfig, t=0., mask=None):
    p = replace(p)  # Make a copy to avoid modifying the input

    if p.loc is None:
        p.loc = force_and_potential(p, cfg=cfg)

    vh = kick(p.vel, p.loc.force() + ext_acc(p, t, cfg), 0.5*dt, mask=mask)
    p.pos = drift(p.pos, vh, dt, mask=mask)
    if p.cpos is not None:
        p.cpos = p.cpos + p.cvel * dt

    p.loc = force_and_potential(p, cfg=cfg)
    p.vel = kick(vh, p.loc.force() + ext_acc(p, t + dt, cfg), 0.5*dt, mask=mask)

    if cfg.centered:
        p = shift_reference_center(p, cfg=cfg)

    return p
timestep.jit = jax.jit(timestep, static_argnames=("cfg",))

def _simulate(p: Particles, tend: float, nsteps: int, cfg: SimConfig, tstart: float = 0.) -> Particles:
    # Make an initial dt=0 step to get the correct initial acceleration
    p = timestep(p, dt=0., cfg=cfg, t=tstart)
    dt = (tend - tstart) / nsteps
    def step(i, carry):
        p, t = carry
        return timestep(p, dt=dt, cfg=cfg, t=t), t + dt

    p, t = jax.lax.fori_loop(0, nsteps, step, (p, tstart))
    return p

def simulate(p: Particles, tend: float, nsteps: int, cfg: SimConfig, tstart: float = 0.) -> Particles:
    @jax.custom_vjp
    def eval(p):
        return _simulate(p, tend, nsteps, cfg, tstart)
    def eval_fwd(p):
        pfin = _simulate(p, tend, nsteps, cfg, tstart)
        return pfin, pfin
    def eval_bwd(p: Particles, gp: jax.Array):
        if cfg.centered:
            warnings.warn(
                "centering makes the simulation poorly reversible and should be avoided in "
                "simmulations with gradients",
                RuntimeWarning,
                stacklevel=2,
            )
        dt = (tend - tstart) / nsteps

        def step(i, carry):
            p, t, gp = carry

            p = timestep(p, dt=-dt, cfg=cfg, t=t)
            _, vjp_fun = jax.vjp(lambda p: timestep(p, dt=dt, cfg=cfg, t=t-dt), p)
            gxp, = vjp_fun(gp)
            return p, t - dt, gxp

        p_prev, t_prev, gp_prev = jax.lax.fori_loop(0, nsteps, step, (p, tend, gp))
        
        return (gp_prev,)
    eval.defvjp(eval_fwd, eval_bwd)
    
    return eval(p)
simulate.jit = jax.jit(simulate, static_argnames=("cfg",))

def clean_particles(p: Particles) -> Particles:
    p = replace(p)  # Make a copy to avoid modifying the input

    p.pos = jnp.asarray(p.pos)
    p.vel = jnp.asarray(p.vel)
    p.mass = jnp.asarray(p.mass)

    if p.loc is None:
        dim = p.pos.shape[-1]
        p.loc = LocalExpansion(jnp.zeros((p.pos.shape[0], dim + 1), dtype=p.pos.dtype), dim=dim)
    
    return p

def simulate_with_outputs(
        p : Particles, 
        tend: float, 
        nout: int, 
        steps_per_output: int,
        cfg: SimConfig,
        tstart: float = 0.
    ) -> Generator[Particles, None, None]:
    """Don't jit this function!"""
    p = clean_particles(p) # This helps avoiding double jit-compilations

    tp0 = time.perf_counter()
    log("Compiling jitted simulation...", level=1, cfg_log=cfg.logging)
    time_dtype = p.pos.dtype
    simulate.jit.lower(p, tend=jnp.asarray(0.1, dtype=time_dtype), nsteps=steps_per_output, cfg=cfg, tstart=jnp.asarray(0.1, dtype=time_dtype)).compile()
    log("Compilation done after {:.2f}s", time.perf_counter()-tp0, level=1, cfg_log=cfg.logging)

    yield tstart, p

    for isnap in range(nout):
        t0 = jnp.asarray(tstart + isnap * (tend/nout), dtype=time_dtype)
        t1 = jnp.asarray(tstart + (isnap+1) * (tend/nout), dtype=time_dtype)
        tpa = time.perf_counter()
        p = simulate.jit(p, tend=t1, nsteps=steps_per_output, cfg=cfg, tstart=t0)
        log("Reached output {} ({:.2f}s for {} steps)",
            isnap+1, time.perf_counter()-tpa, steps_per_output, level=1, cfg_log=cfg.logging)
        yield t1, p
    
    log("Total simulation time: {:.2f}s for {} steps",
        time.perf_counter()-tp0, nout*steps_per_output, level=1, cfg_log=cfg.logging)
