import fmdj
import jax.numpy as jnp
import jax
import pytest

def test_direct_sum_gradient():
    pos = jax.random.uniform(jax.random.PRNGKey(0), (1024, 3))
    mass = jnp.ones_like(pos[:, 0])
    xm = jnp.concatenate([pos, mass[:, None]], axis=1)

    fphi = fmdj.fmm.direct_force_and_potential.jit(xm, softening=1e-2, kahan=True)
    fphi_jax = fmdj.fmm.direct_force_and_potential_jax.jit(pos, mass, softening=1e-2)

    assert fphi == pytest.approx(fphi_jax, rel=1e-4, abs=1e-5)

    def loss(xm):
        fphi = fmdj.fmm.direct_force_and_potential(xm, softening=1e-2, kahan=True)

        return jnp.sum(fphi)

    def loss_jax(xm):
        return jnp.sum(fmdj.fmm.direct_force_and_potential_jax.jit(xm[:,0:3], xm[:,3], softening=1e-2))

    gx1 = jax.grad(loss)(xm)
    gx2 = jax.grad(loss_jax)(xm)
    
    assert gx1 == pytest.approx(gx2, rel=1e-4, abs=1e-5)

def test_sim_com():
    """Tests that gradients with respect to the center of mass work correctly"""

    # Set up a simulation with a uniform acceleration field
    # The center of mass has to move exactly like a particle in the same field
    x = jax.random.uniform(jax.random.PRNGKey(0), (2000,3), minval=1., maxval=2.)
    m = jnp.ones_like(x[:,0]) * 1.
    vel = jnp.zeros_like(x)

    p = fmdj.time_integration.Particles(x, vel, m, cpos=jnp.array([0.,0.,0.]), cvel=jnp.array([0.,0.,0.1]))

    acc = (0.,0.,0.05)

    cfg = fmdj.Config(fmm=None)
    cfg.softening = 0.3
    cfg.external_potential = fmdj.external_potential.UniformAcceleration(acc=acc)

    def loss(p):
        pfin = fmdj.time_integration.simulate.vjp(p, tend=1e2, nsteps=100, cfg=cfg)
        return jnp.sum(jnp.mean(pfin.apos(), axis=0)**2) + jnp.sum(jnp.mean(pfin.avel(), axis=0)**2), pfin
    
    loss_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    (lossval, pfin), pgrad = loss_grad(p)

    def loss_cent(xcom, vcom, t=1e2):
        xfin = xcom + vcom * t + 0.5 * jnp.array(acc) * t**2
        vfin = vcom + jnp.array(acc) * t
        return jnp.sum(xfin**2) + jnp.sum(vfin**2)

    xcom, vcom = jnp.mean(p.apos(), axis=0), jnp.mean(p.avel(), axis=0)
    xcom_grad, vcom_grad = jax.jit(jax.grad(loss_cent, argnums=(0,1)))(xcom, vcom)

    assert pgrad.cpos == pytest.approx(xcom_grad, rel=1e-5)
    assert pgrad.cvel == pytest.approx(vcom_grad, rel=1e-5)

    assert jnp.sum(pgrad.pos, axis=0) == pytest.approx(xcom_grad, rel=1e-5)
    assert jnp.sum(pgrad.vel, axis=0) == pytest.approx(vcom_grad, rel=1e-5)