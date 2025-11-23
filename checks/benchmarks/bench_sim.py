import fmdj
import pytest
from fmdj.config import Config, FMMConfig, LoggingConfig
import jax
import fmdj_ffi as cj
import jax.numpy as jnp
import fmdj.new_tree as nt
import fmdj_ffi.cj_new_tree as cnt
import aegis

@pytest.fixture
def nfw_particles(request):
    npart = request.param if hasattr(request, "param") else 1024*128

    prof = aegis.profiles.NFWProfile(conc=10., r200c=10.)
    pos0, vel0, m = prof.sample_particles(npart, result="pos_vel_m", rpmin=1e-3, ramax=10.)

    part = fmdj.time_integration.Particles(jnp.array(pos0), jnp.array(vel0), jnp.array(m))
    part.cpos = jnp.array((150.,0.,0.))
    part.cvel = jnp.array((0.,prof.vcirc(150.),0.))

    return part

@pytest.fixture
def stripping_cfg():
    cfg = fmdj.config.Config()
    cfg.fmm.alloc_fac_nodes = 3.0
    cfg.logging.level = -1

    host = aegis.profiles.NFWProfile(conc=6., m200c=1e12)
    cfg.external_potential = fmdj.potential.NFWPotential(host.rs, host.rhoc)

    return cfg, host

@pytest.mark.parametrize("nfw_particles", [1024*16, 1024*128, 1024*1024], indirect=True)
def bench_simulate(jax_bench, nfw_particles, stripping_cfg):
    jb = jax_bench(jit_rounds=1, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    cfg, host = stripping_cfg

    p = jb.measure(
        fn=fmdj.time_integration.simulate, fn_jit=fmdj.time_integration.simulate.jit, 
        p=nfw_particles, tend=host.tcirc(150.)*1., nsteps=1000, cfg=cfg
    )[1]

@pytest.mark.parametrize("nfw_particles", [1024*8], indirect=True)
def bench_sim_direct_sum(jax_bench, nfw_particles, stripping_cfg):
    jb = jax_bench(jit_rounds=1, jit_warmup=0, eager_rounds=0, eager_warmup=0)
    
    cfg, host = stripping_cfg
    cfg.fmm = None

    jb.measure(
        fn=fmdj.time_integration.simulate, fn_jit=fmdj.time_integration.simulate.jit, 
        p=nfw_particles, tend=host.tcirc(150.)*1., nsteps=1000, cfg=cfg, tag="sim"
    )

    def loss(p):
        pfin = fmdj.time_integration.simulate.vjp(p, tend=host.tcirc(150.)*1., nsteps=1000, cfg=cfg)
        return jnp.sum(jnp.mean(pfin.apos(), axis=0)**2) + jnp.sum(jnp.mean(pfin.avel(), axis=0)**2)
    
    @jax.jit
    def lossgrad(p):
        return jax.grad(loss)(p)

    jb.measure(
        fn_jit=lossgrad,
        p=nfw_particles, tag="grad"
    )
