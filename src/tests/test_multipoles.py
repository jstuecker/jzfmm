import pytest
import fmdj
import jax
import jax.numpy as jnp
import numpy as np

def setup_particles(N=5555, duplicate=False):
    pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32, minval=-0.5, maxval=0.5)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))

    morton, pos, isort = fmdj.octree.organize_particles(pos0)

    mass = jnp.ones(len(pos), dtype=jnp.float32)
    
    return morton, pos, mass

def test_shift_mp_to_mp_circuit():
    mp0 = np.random.uniform(-0.1, 0.1, (20))

    x1 = np.random.uniform(-1, 1, (3))
    x2 = np.random.uniform(-1, 1, (3))
    x3 = -(x1 + x2)

    mp0 = np.random.uniform(-0.1, 0.1, (20))
    
    mp = mp0
    for dx in [x1, x2, x3]:
        mp = fmdj.multipoles.shift_multipoles(mp, dx, p=3)

    assert np.allclose(mp, mp0, rtol=1e-3), "Shifted local multipoles do not match original multipoles after three shifts."
