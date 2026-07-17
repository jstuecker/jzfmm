from dataclasses import replace
from typing import Generator
import time
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

def find_center(p: Particles, npot: int = 1, nbind: int | None = None):
    if p.loc is None:
        raise ValueError("find_center needs p.loc to contain potentials.")
    if nbind is None:
        nbind = npot

    mass = jnp.broadcast_to(p.mass, p.pos.shape[:-1])

    ipot = jnp.argsort(p.loc.potential())[:npot]
    wpot = mass[ipot]
    cpos = jnp.sum(wpot[:, None] * p.pos[ipot], axis=0) / jnp.sum(wpot)
    cvel0 = jnp.sum(wpot[:, None] * p.vel[ipot], axis=0) / jnp.sum(wpot)

    ebind = p.loc.potential() + 0.5 * jnp.sum((p.vel - cvel0) ** 2, axis=-1)
    ibind = jnp.argsort(ebind)[:nbind]
    wbind = mass[ibind]
    cvel = jnp.sum(wbind[:, None] * p.vel[ibind], axis=0) / jnp.sum(wbind)

    return cpos, cvel
find_center.jit = jax.jit(find_center, static_argnames=("npot", "nbind"))

def ext_acc(p : Particles, t, cfg: SimConfig):
    if cfg.external_potential is None:
        return jnp.zeros_like(p.pos)
    else:
        acc = cfg.external_potential.acceleration(p.pos, t=t, cfg=cfg)

        return acc

def timestep(p : Particles, dt, cfg: SimConfig, t=0., mask=None):
    p = replace(p)  # Make a copy to avoid modifying the input

    assert p.loc is not None

    vh = kick(p.vel, p.loc.force() + ext_acc(p, t, cfg), 0.5*dt, mask=mask)
    p.pos = drift(p.pos, vh, dt, mask=mask)

    p.loc = force_and_potential(p, cfg=cfg)
    p.vel = kick(vh, p.loc.force() + ext_acc(p, t + dt, cfg), 0.5*dt, mask=mask)

    return p
timestep.jit = jax.jit(timestep, static_argnames=("cfg",))

def _simulate(p: Particles, ts: jax.Array, cfg: SimConfig, loc_initialized: bool = False) -> Particles:
    if p.loc is None:
        assert not loc_initialized, "You provided loc_initialized=True, but there is no .loc on particles"
        dim = p.pos.shape[-1]
        loc = LocalExpansion(jnp.zeros(p.pos.shape[:-1] + (dim + 1,), dtype=p.pos.dtype), dim=dim)
        p = replace(p, loc=loc)

    ts = jnp.asarray(ts)
    if not loc_initialized:
        ts = jnp.concatenate((ts[:1], ts))

    def step(i, carry):
        p = carry
        return timestep(p, dt=ts[i+1] - ts[i], cfg=cfg, t=ts[i])

    return jax.lax.fori_loop(0, ts.shape[0] - 1, step, p)

def simulate(
        p: Particles,
        ts: jax.Array,
        cfg: SimConfig,
        loc_initialized: bool = False,
    ) -> Particles:
    ts = jnp.asarray(ts)
    nsteps = ts.shape[0] - 1

    @jax.custom_vjp
    def eval(p):
        return _simulate(p, ts, cfg, loc_initialized=loc_initialized)
    def eval_fwd(p):
        pfin = _simulate(p, ts, cfg, loc_initialized=loc_initialized)
        return pfin, pfin
    def eval_bwd(p: Particles, gp: jax.Array):
        def step(i, carry):
            p, gp = carry
            t0, t1 = ts[nsteps - i - 1], ts[nsteps - i]
            dt = t1 - t0

            p = timestep(p, dt=-dt, cfg=cfg, t=t1)
            _, vjp_fun = jax.vjp(lambda p: timestep(p, dt=dt, cfg=cfg, t=t0), p)
            gxp, = vjp_fun(gp)
            return p, gxp

        p_prev, gp_prev = jax.lax.fori_loop(0, nsteps, step, (p, gp))
        
        return (gp_prev,)
    eval.defvjp(eval_fwd, eval_bwd)
    
    return eval(p)
simulate.jit = jax.jit(simulate, static_argnames=("cfg", "loc_initialized"))

def clean_particles(p: Particles) -> Particles:
    p = replace(p)  # Make a copy to avoid modifying the input

    p.pos = jnp.asarray(p.pos)
    p.vel = jnp.asarray(p.vel)
    p.mass = jnp.asarray(p.mass)

    if p.loc is None:
        dim = p.pos.shape[-1]
        p.loc = LocalExpansion(jnp.zeros(p.pos.shape[:-1] + (dim + 1,), dtype=p.pos.dtype), dim=dim)
    
    return p

def simulate_with_outputs(
        p : Particles, 
        tend: float, 
        nout: int, 
        steps_per_output: int,
        cfg: SimConfig,
        tstart: float = 0.,
        loc_initialized: bool = False,
    ) -> Generator[Particles, None, None]:
    """Don't jit this function!"""
    p = clean_particles(p) # This helps avoiding double jit-compilations

    tp0 = time.perf_counter()
    log("Compiling jitted simulation...", level=1, cfg_log=cfg.logging)
    time_dtype = p.pos.dtype
    ts_compile = jnp.linspace(
        jnp.asarray(0.1, dtype=time_dtype),
        jnp.asarray(0.2, dtype=time_dtype),
        steps_per_output + 1,
        dtype=time_dtype,
    )
    simulate.jit.lower(
        p,
        ts=ts_compile,
        cfg=cfg,
        loc_initialized=loc_initialized,
    ).compile()
    log("Compilation done after {:.2f}s", time.perf_counter()-tp0, level=1, cfg_log=cfg.logging)

    yield tstart, p

    for isnap in range(nout):
        t0 = jnp.asarray(tstart + isnap * (tend/nout), dtype=time_dtype)
        t1 = jnp.asarray(tstart + (isnap+1) * (tend/nout), dtype=time_dtype)
        ts = jnp.linspace(t0, t1, steps_per_output + 1, dtype=time_dtype)
        tpa = time.perf_counter()
        p = simulate.jit(
            p,
            ts=ts,
            cfg=cfg,
            loc_initialized=loc_initialized,
        )
        loc_initialized = True
        log("Reached output {} ({:.2f}s for {} steps)",
            isnap+1, time.perf_counter()-tpa, steps_per_output, level=1, cfg_log=cfg.logging)
        yield t1, p
    
    log("Total simulation time: {:.2f}s for {} steps",
        time.perf_counter()-tp0, nout*steps_per_output, level=1, cfg_log=cfg.logging)
