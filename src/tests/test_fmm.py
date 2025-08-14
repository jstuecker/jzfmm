import pytest
import fmdj
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_dtype_promotion", "strict")

def test_float_types():
    N = 5123
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32) * 0.05
    mass0 = jnp.ones(len(pos0), dtype=pos0.dtype)
    
    phi = fmdj.fmm.fast_multipole_potential(pos0, mass0, use_cj=False, p=3)

    assert phi.dtype == jnp.float32, "Potential should be computed in float32"