import fmdj_jaxonly.jaxonly_fmm
import jax
import jax.numpy as jnp
import fmdj_ffi as cj
import pytest
from fmdj.config import Config, FMMConfig
import fmdj
from fmdj.data import PosMass, TreePlane, InteractionList

import fmdj_jaxonly.jaxonly_multipoles as jmp

def test_expand_interactions():
    nnodes = 3
    npart_per_node = 2

    ilist = InteractionList(
        ispl = jnp.arange(nnodes+1)*2, # [0, 2, 4, 6] each node has 2 interactions
        iother=jnp.array([0, 1, 2, 1, 1, 2]),
        nfilled = 8
    )

    spl = jnp.arange(nnodes)*npart_per_node # [0, 2, 4] each node has 2 particles

    inew = fmdj_jaxonly.jaxonly_fmm.expand_interactions.jit(ilist, spl, size_children=6, size_new_ilist=14)
    
    assert jnp.all(inew.ispl == jnp.array([0,  4,  8, 10, 12, 12, 12]))
    assert jnp.all(inew.iother == jnp.array([0,  1,  2,  3,  
                                             0,  1,  2,  3,  
                                             2,  3,  2,  3, 
                                             14, 14]))


@pytest.fixture
def posz():
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (1024**2,3))
    posz, isort = cj.tree.pos_zorder_sort(pos0)
    return posz

@pytest.fixture
def particlesz(posz):
    return PosMass(posz, jnp.ones(posz.shape[0]))

@pytest.fixture
def tree_hierarchy(particlesz, cfg):
    ths : list[TreePlane] = jax.block_until_ready(fmdj.fmm.build_tree_hierarchy.jit(particlesz, cfg=cfg))
    return ths

@pytest.fixture
def cfg():
    # return Config(tags=("cuda", "base"))
    tcfg = FMMConfig(alloc_fac_nodes=1.2, coarse_fac=2.0, p=2, stop_coarsen=512, ilist_alloc_fac=1024)
    cfg = Config(fmm=tcfg)
    cfg.fmm.opening_angle = 0.85
    return cfg

def test_tree_hierarchy(tree_hierarchy : list[TreePlane]):
    for tplane  in tree_hierarchy:
        assert jnp.sum(tplane.npart) == tplane.tot_npart
        lvls = tplane.lvl[:tplane.nnodes]
        assert jnp.all((lvls >= -100 ) & (lvls < 100))

def test_tree_multipoles(particlesz: PosMass, tree_hierarchy: list[TreePlane], cfg: Config):
    mp_base = jmp.multipoles_from_particles_jax.jit(tree_hierarchy[0], particlesz, cfg=cfg)
    mp_cuda = fmdj.fmm.multipoles_from_particles.jit(tree_hierarchy[0], particlesz, cfg=cfg)
    assert jnp.allclose(tree_hierarchy[0].npart, mp_base.get(0))
    
    for i in range(mp_base.values.shape[1]):
        assert jnp.allclose(mp_base.get(i), mp_cuda.get(i), rtol=1e-3)
    assert jnp.allclose(mp_base.center(), mp_cuda.center(), rtol=1e-6, equal_nan=True)

    mp_coarse = jmp.coarsen_multipoles_jax.jit(mp_base, tree_hierarchy[1], cfg=cfg)
    mp_coarse2 = fmdj.fmm.coarsen_multipoles.jit(mp_cuda, tree_hierarchy[1], cfg=cfg)
    assert jnp.allclose(tree_hierarchy[1].npart, mp_coarse.get(0))
    for i in range(mp_base.values.shape[1]):
        assert jnp.allclose(mp_coarse.get(i), mp_coarse2.get(i), rtol=1e-3, atol=1e-4)

# def test_new_vs_old_tree(particlesz: nt.Particles, tree_hierarchy: list[nt.TreePlane], 
#                          cfg_cuda: Config, fmm_reference : jnp.ndarray):
#     phi_ref = fmm_reference
#     phi = nt.fmm_via_hierarchy_z(tree_hierarchy, particlesz, cfg_cuda)

#     for i in (1000, 1333, 1555, 1777, 5400):
#         print(f"{phi_ref[i]}, {phi[i]}, diff = {phi_ref[i]-phi[i]}, rel diff = {(phi_ref[i]-phi[i])/phi_ref[i]}")

#     assert phi == pytest.approx(phi_ref, rel=0.08)