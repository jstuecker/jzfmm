import fmdj
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
import custom_jax as cj

import pytest

def test_direct_sum_gradient():
    pos = jax.random.uniform(jax.random.PRNGKey(0), (1024, 3))
    mass = jnp.ones_like(pos[:, 0])
    xm = jnp.concatenate([pos, mass[:, None]], axis=1)

    fphi = cj.forces.force_and_potential.jit(xm, softening=1e-2, kahan=True)
    fphi_jax = cj.forces.force_and_potential_pure_jax.jit(pos, mass, softening=1e-2)

    assert fphi == pytest.approx(fphi_jax, rel=1e-4, abs=1e-5)

    def loss(xm):
        fphi = cj.forces.force_and_potential(xm, softening=1e-2, kahan=True)

        return jnp.sum(fphi)

    def loss_jax(xm):
        return jnp.sum(cj.forces.force_and_potential_pure_jax.jit(xm[:,0:3], xm[:,3], softening=1e-2))

    gx1 = jax.grad(loss)(xm)
    gx2 = jax.grad(loss_jax)(xm)
    
    assert gx1 == pytest.approx(gx2, rel=1e-4, abs=1e-5)