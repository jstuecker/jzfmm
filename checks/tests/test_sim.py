import jax
import jax.numpy as jnp

from fmdj.config import DKDLatticeConfig, SimConfig
from fmdj.data import Particles
from fmdj.time_integration import dequantize_particles, quantize_particles, simulate


def test_lattice_reversibility():
    cfg = SimConfig(force=None, integrator=DKDLatticeConfig(dx=1e-4, dv=1e-4, int_dtype=jnp.int32))

    key_pos, key_vel = jax.random.split(jax.random.PRNGKey(0))
    part = Particles(
        pos=0.3 * jax.random.normal(key_pos, (int(1e5), 3), dtype=jnp.float32),
        vel=0.1 * jax.random.normal(key_vel, (int(1e5), 3), dtype=jnp.float32),
        mass=jnp.ones(int(1e5), dtype=jnp.float32) / 1e5,
    )
    part = dequantize_particles(quantize_particles(part, cfg.integrator), cfg.integrator)

    ts = jnp.linspace(0.0, 1.0, 17, dtype=jnp.float32)
    part_fwd = simulate.jit(part, ts=ts, cfg=cfg)
    part_bwd = simulate.jit(part_fwd, ts=ts[::-1], cfg=cfg)

    assert jnp.array_equal(part_bwd.pos, part.pos)
    assert jnp.array_equal(part_bwd.vel, part.vel)
