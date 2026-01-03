import fmdj
import pytest
import jax
import jax.numpy as jnp
import aegis
from dataclasses import replace


@pytest.fixture
def stripping_cfg():
    cfg = fmdj.config.Config()
    cfg.logging.level = -1

    host = aegis.profiles.NFWProfile(conc=6., m200c=1e12)
    cfg.external_potential = fmdj.external_potential.NFWPotential(host.rs, host.rhoc)

    return cfg, host

@pytest.mark.parametrize("npart", [1024*16, 1024*128, 1024*1024], indirect=True)
def bench_simulate(jax_bench, particles_nfw: fmdj.data.Particles, stripping_cfg):
    jb = jax_bench(jit_rounds=1, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    cfg, host = stripping_cfg

    jb.measure(
        fn=fmdj.time_integration.simulate, fn_jit=fmdj.time_integration.simulate.jit, 
        p=particles_nfw, tend=host.tcirc(150.)*1., nsteps=200, cfg=cfg
    )

@pytest.mark.parametrize("npart", [1024*8], indirect=True)
def bench_sim_direct_sum(jax_bench, particles_nfw, stripping_cfg):
    jb = jax_bench(jit_rounds=1, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    
    cfg, host = stripping_cfg
    cfg = replace(cfg, fmm=None)

    jb.measure(
        fn=fmdj.time_integration.simulate, fn_jit=fmdj.time_integration.simulate.jit, 
        p=particles_nfw, tend=host.tcirc(150.)*1., nsteps=1000, cfg=cfg, tag="sim"
    )

    def loss(p):
        pfin = fmdj.time_integration.simulate.vjp(p, tend=host.tcirc(150.)*1., nsteps=1000, cfg=cfg)
        return jnp.sum(jnp.mean(pfin.apos(), axis=0)**2) + jnp.sum(jnp.mean(pfin.avel(), axis=0)**2)
    
    @jax.jit
    def lossgrad(p):
        return jax.grad(loss)(p)

    jb.measure(
        fn_jit=lossgrad,
        p=particles_nfw, tag="grad"
    )