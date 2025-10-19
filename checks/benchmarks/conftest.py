import jax
import jax.numpy as jnp
import pytest
import fmdj

def get_particles(N = 1024*1024):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
    pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
    mass = jnp.ones(N, dtype=jnp.float32)
    
    return jax.block_until_ready((pos0, mass))

@pytest.fixture
def particles(): return get_particles(1024*1024)