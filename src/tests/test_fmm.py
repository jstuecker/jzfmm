import pytest
import fmdj
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_dtype_promotion", "strict")

def test_float_types():
    N = 5123
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N,3)) * 0.05
    mass0 = jnp.ones(len(pos0))
    
    phi32 = fmdj.fmm.fast_multipole_potential(pos0.astype(jnp.float32), mass0.astype(jnp.float32), use_cj=False, p=2)

    assert phi32.dtype == jnp.float32, "Potential should be computed in float32"

    phi64 = fmdj.fmm.fast_multipole_potential(pos0.astype(jnp.float64), mass0.astype(jnp.float64), use_cj=False, p=2)

    assert phi64.dtype == jnp.float64, "Potential should be computed in float64"