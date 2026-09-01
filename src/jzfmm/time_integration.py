from dataclasses import replace
from typing import Generator
import time
import jax.numpy as jnp
import jax

from jztree.tools import log
from .data import Particles, LocalExpansion
from .config import DirectSummationConfig, DKDConfig, DKDLatticeConfig, FMMConfig, KDKConfig, SimConfig
from .fmm import direct_summation, fast_multipole_method

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

def find_center(p: Particles, npot: int = 50, nbind: int | None = None):
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

def _ext_acc(p : Particles, t, cfg: SimConfig):
    if cfg.external_potential is None:
        return jnp.zeros_like(p.pos)
    return cfg.external_potential.acceleration(p.pos, t=t, cfg=cfg)

def quantize_particles(p: Particles, integrator: DKDLatticeConfig) -> Particles:
    dtype = jnp.result_type(p.mass, jnp.float32)
    with jax.enable_x64(integrator.int_dtype == jnp.int64):
        pos = jnp.rint(p.pos / jnp.asarray(integrator.dx, dtype=dtype)).astype(integrator.int_dtype)
        vel = jnp.rint(p.vel / jnp.asarray(integrator.dv, dtype=dtype)).astype(integrator.int_dtype)
    return replace(
        p,
        pos=pos,
        vel=vel,
    )

def dequantize_particles(p: Particles, integrator: DKDLatticeConfig) -> Particles:
    dtype = jnp.result_type(p.mass, jnp.float32)
    with jax.enable_x64(integrator.int_dtype == jnp.int64):
        pos = p.pos.astype(dtype) * jnp.asarray(integrator.dx, dtype=dtype)
        vel = p.vel.astype(dtype) * jnp.asarray(integrator.dv, dtype=dtype)
    return replace(
        p,
        pos=pos,
        vel=vel,
    )

def _timestep_kdk(p : Particles, dt, cfg: SimConfig, t=0.):
    p = replace(p)  # Make a copy to avoid modifying the input

    assert p.loc is not None, "need to have previous acceleration for KDK"

    vh = p.vel + (p.loc.force() + _ext_acc(p, t, cfg)) * (0.5 * dt)
    p.pos = p.pos + vh * dt

    p.loc = force_and_potential(p, cfg=cfg)
    p.vel = vh + (p.loc.force() + _ext_acc(p, t + dt, cfg)) * (0.5 * dt)

    return p

def _timestep_dkd(p : Particles, dt, cfg: SimConfig, t=0.):
    p = replace(p)  # Make a copy to avoid modifying the input

    assert p.loc is None, "Should not have p.loc on particles for DKD, it is internal and temporary"

    p.pos = p.pos + p.vel * (0.5 * dt)
    loc = force_and_potential(p, cfg=cfg)
    p.vel = p.vel + (loc.force() + _ext_acc(p, t + 0.5 * dt, cfg)) * dt
    p.pos = p.pos + p.vel * (0.5 * dt)

    return p

def _timestep_dkd_lattice(p : Particles, dt, cfg: SimConfig, t=0.):
    p = replace(p)  # Make a copy to avoid modifying the input

    assert p.loc is None, "Should not have p.loc on particles for DKD, it is internal and temporary"
    assert jnp.issubdtype(p.pos.dtype, jnp.integer)
    assert jnp.issubdtype(p.vel.dtype, jnp.integer)

    dx, dv, int_dtype = cfg.integrator.dx, cfg.integrator.dv, cfg.integrator.int_dtype

    with jax.enable_x64(int_dtype == jnp.int64):
        p.pos = p.pos + jnp.rint(p.vel * (0.5 * dt * dv / dx)).astype(int_dtype)

    p_float = dequantize_particles(p, cfg.integrator)
    loc = force_and_potential(p_float, cfg=cfg)
    acc = loc.force() + _ext_acc(p_float, t + 0.5 * dt, cfg)
    with jax.enable_x64(int_dtype == jnp.int64):
        p.vel = p.vel + jnp.rint(dt * acc / dv).astype(int_dtype)

    with jax.enable_x64(int_dtype == jnp.int64):
        p.pos = p.pos + jnp.rint(p.vel * (0.5 * dt * dv / dx)).astype(int_dtype)

    return p

def _reverse_timestep_dkd_vjp(p_next: Particles, gp_next: Particles, dt, cfg: SimConfig, tend=0.):
    xmid = p_next.pos - 0.5 * dt * p_next.vel
    p_mid = replace(p_next, pos=xmid, loc=None)

    def mid_acc(p):
        return force_and_potential(p, cfg=cfg).force() + _ext_acc(p, tend - 0.5 * dt, cfg)

    acc_mid, acc_vjp = jax.vjp(mid_acc, p_mid)
    v_prev = p_next.vel - dt * acc_mid
    p_prev = replace(p_next, pos=xmid - 0.5 * dt * v_prev, vel=v_prev, loc=None)

    gv = gp_next.vel + 0.5 * dt * gp_next.pos
    gmid, = acc_vjp(dt * gv) # dt * gv corresponds to the gradient w.r.t. to the output force

    gp_prev = replace(
        gp_next,
        pos=gp_next.pos + gmid.pos,
        vel=gv + 0.5 * dt * (gp_next.pos + gmid.pos),
        mass=gp_next.mass + gmid.mass,
        loc=None,
    )

    return p_prev, gp_prev

def _reverse_timestep_dkd_lattice_vjp(p_next: Particles, gp_next: Particles, dt, cfg: SimConfig, tend=0.):
    p_prev = replace(p_next)
    dx, dv, int_dtype = cfg.integrator.dx, cfg.integrator.dv, cfg.integrator.int_dtype

    with jax.enable_x64(int_dtype == jnp.int64):
        p_prev.pos = p_prev.pos - jnp.rint(p_prev.vel * (0.5 * dt * dv / dx)).astype(int_dtype)

    p_mid = dequantize_particles(p_prev, cfg.integrator)

    def mid_acc(p):
        return force_and_potential(p, cfg=cfg).force() + _ext_acc(p, tend - 0.5 * dt, cfg)

    acc_mid, acc_vjp = jax.vjp(mid_acc, p_mid)
    with jax.enable_x64(int_dtype == jnp.int64):
        p_prev.vel = p_prev.vel - jnp.rint(dt * acc_mid / dv).astype(int_dtype)
        p_prev.pos = p_prev.pos - jnp.rint(p_prev.vel * (0.5 * dt * dv / dx)).astype(int_dtype)

    gv = gp_next.vel + 0.5 * dt * gp_next.pos
    gmid, = acc_vjp(dt * gv) # dt * gv corresponds to the gradient w.r.t. to the output force

    gp_prev = replace(
        gp_next,
        pos=gp_next.pos + gmid.pos,
        vel=gv + 0.5 * dt * (gp_next.pos + gmid.pos),
        mass=gp_next.mass + gmid.mass,
        loc=None,
    )

    return p_prev, gp_prev

def timestep(p : Particles, dt, cfg: SimConfig, t=0.):
    if isinstance(cfg.integrator, KDKConfig):
        return _timestep_kdk(p, dt=dt, cfg=cfg, t=t)
    if isinstance(cfg.integrator, DKDConfig):
        return _timestep_dkd(p, dt=dt, cfg=cfg, t=t)
    if isinstance(cfg.integrator, DKDLatticeConfig):
        return _timestep_dkd_lattice(p, dt=dt, cfg=cfg, t=t)
    raise TypeError(f"Unknown integrator config {cfg.integrator!r}. Expected KDKConfig, DKDConfig, or DKDLatticeConfig.")
timestep.jit = jax.jit(timestep, static_argnames=("cfg",))

def _simulate(p: Particles, ts: jax.Array, cfg: SimConfig) -> Particles:
    ts = jnp.asarray(ts)
    if isinstance(cfg.integrator, KDKConfig):
        if p.loc is None:
            # prepend a zero-size time-step to initialize p.loc properly.
            dim = p.pos.shape[-1]
            loc = LocalExpansion(jnp.zeros(p.pos.shape[:-1] + (dim + 1,), dtype=p.pos.dtype), dim=dim)
            p = replace(p, loc=loc)
            ts = jnp.concatenate((ts[:1], ts))
    elif isinstance(cfg.integrator, DKDConfig):
        assert p.loc is None
    elif isinstance(cfg.integrator, DKDLatticeConfig):
        assert p.loc is None
        p = quantize_particles(p, cfg.integrator)
    else:
        raise TypeError(f"Unknown integrator config {cfg.integrator!r}. Expected KDKConfig, DKDConfig, or DKDLatticeConfig.")

    def step(i, carry):
        p = carry
        return timestep(p, dt=ts[i+1] - ts[i], cfg=cfg, t=ts[i])

    p = jax.lax.fori_loop(0, ts.shape[0] - 1, step, p)
    if isinstance(cfg.integrator, DKDLatticeConfig):
        p = dequantize_particles(p, cfg.integrator)
    return p

def simulate(
        p: Particles,
        ts: jax.Array,
        cfg: SimConfig,
    ) -> Particles:
    ts = jnp.asarray(ts)
    nsteps = ts.shape[0] - 1

    @jax.custom_vjp
    def eval(p):
        return _simulate(p, ts, cfg)
    def eval_fwd(p):
        pfin = _simulate(p, ts, cfg)
        return pfin, pfin
    def eval_bwd(p: Particles, gp: jax.Array):
        assert isinstance(cfg.integrator, (DKDConfig, DKDLatticeConfig)), (
            f"Differentiation through simulate is only supported with DKDConfig or DKDLatticeConfig, "
            f"but cfg.integrator is {cfg.integrator!r}."
        )
        if isinstance(cfg.integrator, DKDLatticeConfig):
            p = quantize_particles(p, cfg.integrator)

        def step(i, carry):
            p, gp = carry
            t0, t1 = ts[nsteps - i - 1], ts[nsteps - i]
            dt = t1 - t0

            if isinstance(cfg.integrator, DKDConfig):
                return _reverse_timestep_dkd_vjp(p, gp, dt=dt, cfg=cfg, tend=t1)
            return _reverse_timestep_dkd_lattice_vjp(p, gp, dt=dt, cfg=cfg, tend=t1)

        p_prev, gp_prev = jax.lax.fori_loop(0, nsteps, step, (p, gp))
        
        return (gp_prev,)
    eval.defvjp(eval_fwd, eval_bwd)
    
    return eval(p)
simulate.jit = jax.jit(simulate, static_argnames=("cfg",))

def simulate_with_outputs(
        p : Particles, 
        tend: float, 
        nout: int, 
        steps_per_output: int,
        cfg: SimConfig,
        tstart: float = 0.,
    ) -> Generator[Particles, None, None]:
    """Don't jit this function!"""
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
        )
        log("Reached output {} ({:.2f}s for {} steps)",
            isnap+1, time.perf_counter()-tpa, steps_per_output, level=1, cfg_log=cfg.logging)
        yield t1, p
    
    log("Total simulation time: {:.2f}s for {} steps",
        time.perf_counter()-tp0, nout*steps_per_output, level=1, cfg_log=cfg.logging)
