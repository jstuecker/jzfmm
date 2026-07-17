import pytest
import jax
import jax.numpy as jnp
import aegis
from dataclasses import replace
from fmdj import DirectSummationConfig, SimConfig
from fmdj.external_potential import NFWPotential
from fmdj.data import Particles
from fmdj.time_integration import simulate


@pytest.fixture
def stripping_cfg():
    cfg = SimConfig()
    cfg.logging.level = -1

    host = aegis.profiles.NFWProfile(conc=6., m200c=1e12)
    cfg.external_potential = NFWPotential(host.rs, host.rhoc)

    return cfg, host

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("npart", [1024*128, 1024*1024], indirect=True)
def bench_simulate(jax_bench, particles_nfw: Particles, stripping_cfg):
    jb = jax_bench(jit_rounds=1, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    cfg, host = stripping_cfg
    ts = jnp.linspace(0.0, host.tcirc(150.)*1., 201, dtype=particles_nfw.pos.dtype)

    jb.measure(
        fn=simulate, fn_jit=simulate.jit, 
        p=particles_nfw, ts=ts, cfg=cfg
    )

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("npart", [1024*128, 1024*1024], indirect=True)
def bench_sim_grad(jax_bench, particles_nfw: Particles, stripping_cfg):
    jb = jax_bench(jit_rounds=1, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    cfg, host = stripping_cfg
    ts = jnp.linspace(0.0, host.tcirc(150.)*1., 201, dtype=particles_nfw.pos.dtype)

    def loss(p):
        pfin = simulate(p, ts=ts, cfg=cfg)
        return jnp.mean(jnp.sum(pfin.pos**2, axis=-1))

    @jax.jit
    def lossgrad(p):
        return jax.grad(loss)(p)

    jb.measure(
        fn_jit=lossgrad,
        p=particles_nfw,
    )

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("npart", [1024*8], indirect=True)
def bench_sim_direct_sum(jax_bench, particles_nfw, stripping_cfg):
    jb = jax_bench(jit_rounds=1, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    
    cfg, host = stripping_cfg
    cfg = replace(cfg, force=DirectSummationConfig())
    ts = jnp.linspace(0.0, host.tcirc(150.)*1., 1001, dtype=particles_nfw.pos.dtype)

    jb.measure(
        fn=simulate, fn_jit=simulate.jit, 
        p=particles_nfw, ts=ts, cfg=cfg, tag="sim"
    )

    def loss(p):
        pfin = simulate(p, ts=ts, cfg=cfg)
        return jnp.sum(jnp.mean(pfin.pos, axis=0)**2) + jnp.sum(jnp.mean(pfin.vel, axis=0)**2)
    
    @jax.jit
    def lossgrad(p):
        return jax.grad(loss)(p)

    jb.measure(
        fn_jit=lossgrad,
        p=particles_nfw, tag="grad"
    )
