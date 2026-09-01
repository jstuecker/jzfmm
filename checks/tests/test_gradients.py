import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import jax.numpy as jnp
import jax
import pytest
from dataclasses import replace
from jax.test_util import check_grads

from jztree.data import PosLvl
from jztree.tree import _dense_interaction_list
from jztree.config import TreeConfig
from jztree_utils import ics

from jzfmm.config import DirectSummationConfig, FMMConfig, OpeningByAngle, PlummerKernel, SimConfig
from jzfmm.data import PosMass
from jzfmm.multipoles import summarize_multipoles, build_multipole_hierarchy, _fmm_node_to_child
from jzfmm.fmm import _evaluate_node_node_fmm, _fmm_dual_walk
from jzfmm.fmm import _leaf_leaf_summation, direct_summation, fast_multipole_method
from jzfmm.external_potential import UniformAcceleration
from jzfmm.time_integration import simulate
    
def my_check_gradient(f, x, epsrel=1e-4, rtol=5e-3, atol=0.):
    dx = jnp.std(x, axis=0, keepdims=True) * epsrel
    dx = dx * jax.random.normal(jax.random.PRNGKey(0), x.shape)
    
    df1 = f(x + dx) - f(x)
    df2 = jnp.sum(jax.grad(f)(x) * dx)

    # print("Finite diff:", df1)
    # print("JAX grad  :", df2)
    # print(f"Relative diff: {jnp.abs(df1 - df2) / (jnp.abs(df1) + jnp.abs(df2) + 1e-30):.2e}")

    assert df2 == pytest.approx(df1, rel=rtol, abs=atol)

@pytest.mark.skip_in_quick
def test_m2m_gradients(pos_mass_z, tree_hierarchy):
    cfg_fmm = FMMConfig()
    spl = tree_hierarchy.splits_leaf_to_part()
    node = tree_hierarchy.poslvl(0)
    child = PosLvl(pos=pos_mass_z.pos, lvl=jnp.zeros(len(pos_mass_z.pos), dtype=jnp.int32))

    def m2m(x,m):
        return summarize_multipoles(spl, m, node, replace(child, pos=x), cfg_fmm=cfg_fmm)

    check_grads(lambda m: m2m(pos_mass_z.pos, m), (pos_mass_z.mass,), order=1, modes=("rev",), eps=1e-3)
    check_grads(lambda x: m2m(x, pos_mass_z.mass), (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-3)
    check_grads(m2m, (pos_mass_z.pos, pos_mass_z.mass), order=1, modes=("rev",), eps=1e-3)

@pytest.mark.skip_in_quick
def test_l2l_gradients(pos_mass_z, tree_hierarchy):
    th = tree_hierarchy
    cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=1e-1))
    mph = build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg_fmm=cfg_fmm)
    loc, ilist = _fmm_dual_walk.jit(th, mph, cfg_fmm=cfg_fmm)

    ispl = th.splits_leaf_to_part()
    node = th.poslvl(0)
    child = PosLvl(pos=pos_mass_z.pos, lvl=jnp.zeros(len(pos_mass_z.pos), dtype=jnp.int32))
    def l2l(x,loc):
        return _fmm_node_to_child(ispl, loc, node, replace(child, pos=x), cfg_fmm=cfg_fmm, pout=1)

    loc = loc.at[:,1:4].set(0.)
    check_grads(lambda x: l2l(x, loc), (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-2)
    # For multipole gradients we need to use smarter finit difference steps than jax's default:
    my_check_gradient(lambda l: l2l(pos_mass_z.pos, l).sum(), loc, epsrel=5e-2)

@pytest.mark.skip_in_quick
def test_fmm_node_gradients(pos_mass_z, tree_hierarchy):
    cfg_fmm = FMMConfig(p=4, kernel=PlummerKernel(softening=1e-1))

    def f(pos):
        pm = PosMass(pos=pos, mass=pos_mass_z.mass)
        return _evaluate_node_node_fmm(pm, tree_hierarchy, cfg_fmm=cfg_fmm)[0]
    check_grads(f, (pos_mass_z.pos,), order=1, modes=("rev",), eps=1e-2)

    def f(mass):
        pm = PosMass(pos=pos_mass_z.pos, mass=mass)
        return _evaluate_node_node_fmm(pm, tree_hierarchy, cfg_fmm=cfg_fmm)[0]
    check_grads(f, (pos_mass_z.mass,), order=1, modes=("rev",), eps=1e-1)

@pytest.mark.parametrize("mode", ["fmm", "direct"])
@pytest.mark.parametrize("npart", [1024], indirect=True)
def test_sim_com(particles_blob, mode):
    """Tests that gradients with respect to the center of mass work correctly"""
    p = replace(particles_blob, vel=particles_blob.vel + jnp.array([0.,0.,0.1]))

    acc = (0.,0.,0.05)

    cfg_fmm = FMMConfig(
        kernel=PlummerKernel(softening=0.3),
        tree=TreeConfig(mass_centered=False, alloc_fac_nodes=2.0),
        alloc_fac_ilist=256.,
    )
    cfg = SimConfig(force=cfg_fmm)
    cfg.external_potential = UniformAcceleration(acc=acc)
    if mode == "direct":
        cfg.force = DirectSummationConfig(kernel=PlummerKernel(softening=0.3))

    def loss(p):
        ts = jnp.linspace(0.0, 1e2, 101, dtype=p.pos.dtype)
        pfin = simulate(p, ts=ts, cfg=cfg)
        return jnp.sum(jnp.mean(pfin.pos, axis=0)**2) + jnp.sum(jnp.mean(pfin.vel, axis=0)**2), pfin
    
    loss_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    (lossval, pfin), pgrad = loss_grad(p)

    def loss_cent(xcom, vcom, t=1e2):
        xfin = xcom + vcom * t + 0.5 * jnp.array(acc) * t**2
        vfin = vcom + jnp.array(acc) * t
        return jnp.sum(xfin**2) + jnp.sum(vfin**2)

    xcom, vcom = jnp.mean(p.pos, axis=0), jnp.mean(p.vel, axis=0)
    xcom_grad, vcom_grad = jax.jit(jax.grad(loss_cent, argnums=(0,1)))(xcom, vcom)

    assert jnp.sum(pgrad.pos, axis=0) == pytest.approx(xcom_grad, rel=3e-4)
    assert jnp.sum(pgrad.vel, axis=0) == pytest.approx(vcom_grad, rel=3e-4)

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("dim", (2,3))
def test_force_gradients(dim):
    npart = 1024*8
    part = ics.gaussian_particles(npart, dim=dim, total_mass=npart*1.)
    part.num = None # currently causes some problems with gradients
    part.num_total = None # currently causes some problems with gradients
    part.mass = jnp.broadcast_to(part.mass, part.pos.shape[:-1])
    cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=0.05), p=4, kahan_summation=True, opening=OpeningByAngle(theta=0.8))
    
    ispl = jnp.arange(part.pos.shape[0]//32 + 1, dtype=jnp.int32) * 32
    ilist = _dense_interaction_list.jit(len(ispl)-1, len(ispl)-1, (len(ispl)-1)**2)

    cfg_direct = DirectSummationConfig(kernel=cfg_fmm.kernel, kahan_summation=True)
    fphi1 = direct_summation.jit(part, cfg_direct=cfg_direct).values
    fphi2 = _leaf_leaf_summation.jit(part, ispl, ilist, cfg_fmm)
    fphi3 = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values

    abstol = float(jnp.std(fphi1) * 2e-2)

    assert fphi2 == pytest.approx(fphi1, abs=abstol*1e-2)
    assert fphi3 == pytest.approx(fphi1, abs=abstol)

    def f1(part): return _leaf_leaf_summation(part, ispl, ilist, cfg_fmm=cfg_fmm).sum()
    def f2(part): return direct_summation(part, cfg_direct=cfg_direct).values.sum()
    def f3(part): return fast_multipole_method(part, cfg_fmm=cfg_fmm).values.sum()
    
    gposm1 = jax.jit(jax.grad(f1, allow_int=True))(part)
    gposm2 = jax.jit(jax.grad(f2, allow_int=True))(part)
    gposm3 = jax.jit(jax.grad(f3, allow_int=True))(part)

    abstol_pos = float(jnp.std(gposm2.pos) * 4e-2)
    abstol_mass = float(jnp.std(gposm2.mass) * 3e-2)

    assert gposm2.pos == pytest.approx(gposm1.pos, abs=abstol_pos*1e-2)
    assert gposm2.mass == pytest.approx(gposm1.mass, abs=abstol_mass*1e-2)

    assert gposm3.pos == pytest.approx(gposm1.pos, abs=abstol_pos)
    assert gposm3.mass == pytest.approx(gposm1.mass, abs=abstol_mass)
