import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

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

def test_m2m_gradients(pos_mass_z, tree_planes, cfg):
    th = tree_planes
    # mph = fmdj.multipoles.build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg)

    def m2m(x,m):
        return fmdj.multipoles.summarize_multipoles(th[0].ispl, m, th[0].center(), x, cfg=cfg)

    check_grads(lambda m: m2m(pos_mass_z.pos, m), (pos_mass_z.mass,), order=1, modes=("rev",), eps=1e-3)
    check_grads(lambda x: m2m(x, pos_mass_z.mass), (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-3)
    check_grads(m2m, (pos_mass_z.pos, pos_mass_z.mass), order=1, modes=("rev",), eps=1e-3)

def test_l2l_gradients(pos_mass_z, tree_planes, cfg):
    th = tree_planes
    cfg = replace(cfg, softening=1e-1)
    mph = fmdj.multipoles.build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg=cfg)
    loc, ilist = fmdj.fmm.evaluate_interaction_hierarchy.jit(th, mph, cfg)

    def l2l(x,loc):
        return fmdj.multipoles.shift_local_to_children(th[0].ispl, loc, th[0].center(), x, cfg=cfg, pout=1)

    loc = loc.at[:,1:4].set(0.)
    check_grads(lambda x: l2l(x, loc), (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-2)
    # For multipole gradients we need to use smarter finit difference steps than jax's default:
    my_check_gradient(lambda l: l2l(pos_mass_z.pos, l).sum(), loc, epsrel=5e-2)

def test_fmm_node_gradients(pos_mass_z, tree_planes, cfg):
    cfg_fmm = replace(cfg.fmm, p=4)
    cfg = replace(cfg, softening=1e-1, fmm=cfg_fmm)

    def f(pos):
        pm = fmdj.data.PosMass(pos, pos_mass_z.mass)
        return fmdj.fmm.evaluate_node_node_fmm(pm, tree_planes, cfg=cfg)[0]
    check_grads(f, (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-2)

    def f(mass):
        pm = fmdj.data.PosMass(pos_mass_z.pos, mass)
        return fmdj.fmm.evaluate_node_node_fmm(pm, tree_planes, cfg=cfg)[0]
    check_grads(f, (pos_mass_z.mass,), order=1, modes=("rev",), eps=1e-1)

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

@pytest.mark.parametrize("npart", [1024*8], indirect=True)
def test_force_gradients(pos_mass: fmdj.data.PosMass):
    part = pos_mass
    fmmcfg = fmdj.config.FMMConfig(p=4, kahan_summation=True, opening_angle=0.8)
    cfg = fmdj.Config(softening=0.05, fmm=fmmcfg)
    
    ispl = jnp.arange(part.pos.shape[0]//32 + 1, dtype=jnp.int32) * 32
    ilist = fmdj.ztree.dense_interaction_list.jit(len(ispl)-1, len(ispl)-1, (len(ispl)-1)**2)

    fphi1 = fmdj.fmm.direct_force_and_potential.jit(part, softening=cfg.softening, kahan=True)
    fphi2 = fmdj.fmm.grouped_force_and_pot.jit(part, ispl, ilist, cfg)
    fphi3 = fmdj.fmm.fast_multipole_method.jit(part, cfg=cfg).values / cfg.G()

    abstol = float(jnp.std(fphi1) * 1e-2)

    assert fphi2 == pytest.approx(fphi1, abs=abstol*1e-2)
    assert fphi3 == pytest.approx(fphi1, abs=abstol)

    def f1(part): return fmdj.fmm.grouped_force_and_pot(part, ispl, ilist, cfg=cfg).sum()
    def f2(part): return fmdj.fmm.direct_force_and_potential(part, softening=cfg.softening, kahan=True).sum()
    def f3(part): return fmdj.fmm.fast_multipole_method(part, cfg=cfg).values.sum() / cfg.G()
    
    gposm1 = jax.jit(jax.grad(f1))(part)
    gposm2 = jax.jit(jax.grad(f2))(part)
    gposm3 = jax.jit(jax.grad(f3))(part)

    abstol_pos = float(jnp.std(gposm2.pos) * 2e-2)
    abstol_mass = float(jnp.std(gposm2.mass) * 1e-2)

    assert gposm2.pos == pytest.approx(gposm1.pos, abs=abstol_pos*1e-2)
    assert gposm2.mass == pytest.approx(gposm1.mass, abs=abstol_mass*1e-2)

    assert gposm3.pos == pytest.approx(gposm1.pos, abs=abstol_pos)
    assert gposm3.mass == pytest.approx(gposm1.mass, abs=abstol_mass)