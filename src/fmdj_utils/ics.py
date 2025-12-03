import numpy as np
import jax
import jax.numpy as jnp
import fmdj

def gaussian_blob(N, scale=1.0, mass=1., seed=0, zsort=False):
    pos = jax.random.normal(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32) * scale
    if zsort:
        pos, isort = fmdj.ztree.pos_zorder_sort(pos)
    mass0 = jnp.ones(len(pos), dtype=pos.dtype) * (mass/N)
    return fmdj.data.PosMass(pos, mass0)