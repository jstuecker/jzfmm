import fmdj
import jax.numpy as jnp
import jax
import pytest
from dataclasses import replace

from jax.test_util import check_grads

def my_check_gradient(f, x, epsrel=1e-4, rtol=5e-3, atol=0.):
    dx = jnp.std(x, axis=0, keepdims=True) * epsrel
    dx = dx * jax.random.normal(jax.random.PRNGKey(0), x.shape)
    
    df1 = f(x + dx) - f(x)
    df2 = jnp.sum(jax.grad(f)(x) * dx)

    # print("Finite diff:", df1)
    # print("JAX grad  :", df2)
    # print(f"Relative diff: {jnp.abs(df1 - df2) / (jnp.abs(df1) + jnp.abs(df2) + 1e-30):.2e}")

    assert df2 == pytest.approx(df1, rel=rtol, abs=atol)

def test_m2m_gradients(pos_mass_z, tree_hierarchy, cfg):
    th = tree_hierarchy
    # mph = fmdj.multipoles.build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg)

    def m2m(x,m):
        return fmdj.multipoles.summarize_multipoles(th[0].ispl, m, th[0].center(), x, cfg=cfg)

    check_grads(lambda m: m2m(pos_mass_z.pos, m), (pos_mass_z.mass,), order=1, modes=("rev",), eps=1e-3)
    check_grads(lambda x: m2m(x, pos_mass_z.mass), (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-3)
    check_grads(m2m, (pos_mass_z.pos, pos_mass_z.mass), order=1, modes=("rev",), eps=1e-3)

def test_l2l_gradients(pos_mass_z, tree_hierarchy, cfg):
    th = tree_hierarchy
    cfg = replace(cfg, softening=1e-1)
    mph = fmdj.multipoles.build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg=cfg)
    loc, ilist = fmdj.fmm.evaluate_interaction_hierarchy.jit(th, mph, cfg)

    def l2l(x,loc):
        return fmdj.multipoles.shift_local_to_children(th[0].ispl, loc, th[0].center(), x, cfg=cfg, pout=1)

    loc = loc.at[:,1:4].set(0.)
    check_grads(lambda x: l2l(x, loc), (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-2)
    # For multipole gradients we need to use smarter finit difference steps than jax's default:
    my_check_gradient(lambda l: l2l(pos_mass_z.pos, l).sum(), loc, epsrel=5e-2)

def test_fmm_node_gradients(pos_mass_z, tree_hierarchy, cfg):
    cfg_fmm = replace(cfg.fmm, p=4, multipoles_around_com=True)
    cfg = replace(cfg, softening=1e-1, fmm=cfg_fmm)

    def f(pos):
        pm = fmdj.data.PosMass(pos, pos_mass_z.mass)
        return fmdj.fmm.evaluate_node_node_fmm(pm, tree_hierarchy, cfg=cfg)[0]
    check_grads(f, (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-2)

    def f(mass):
        pm = fmdj.data.PosMass(pos_mass_z.pos, mass)
        return fmdj.fmm.evaluate_node_node_fmm(pm, tree_hierarchy, cfg=cfg)[0]
    check_grads(f, (pos_mass_z.mass,), order=1, modes=("rev",), eps=1e-1)

@pytest.mark.parametrize("npart", [1024], indirect=True)
def test_direct_sum_gradient(pos_mass_z: fmdj.data.PosMass):
    loc = fmdj.fmm.direct_force_and_potential.jit(pos_mass_z.posm(), softening=1e-2, kahan=True)
    loc = fmdj.data.LocalExpansion(loc)
    fphi_jax = fmdj.fmm.direct_force_and_potential_jax.jit(pos_mass_z.pos, pos_mass_z.mass, softening=1e-2)

    assert loc.fphi() == pytest.approx(fphi_jax, rel=1e-4, abs=1e-5)

    def loss(xm):
        loc = fmdj.fmm.direct_force_and_potential(xm, softening=1e-2, kahan=True)

        return jnp.sum(fmdj.data.LocalExpansion(loc).fphi())

    def loss_jax(xm):
        return jnp.sum(fmdj.fmm.direct_force_and_potential_jax.jit(xm[:,0:3], xm[:,3], softening=1e-2))

    gx1 = jax.grad(loss)(pos_mass_z.posm())
    gx2 = jax.grad(loss_jax)(pos_mass_z.posm())
    
    assert gx1 == pytest.approx(gx2, rel=1e-3, abs=1e-5)

@pytest.mark.parametrize("npart", [1024], indirect=True)
def test_sim_com(particles_blob):
    """Tests that gradients with respect to the center of mass work correctly"""
    p = replace(particles_blob, cvel=jnp.array([0.,0.,0.1]))

    acc = (0.,0.,0.05)

    cfg = fmdj.Config(fmm=None, softening=0.3)
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