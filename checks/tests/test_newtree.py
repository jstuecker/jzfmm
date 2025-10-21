import fmdj.new_tree as nt
import jax
import jax.numpy as jnp
import custom_jax as cj
import pytest
from custom_jax.cj_new_tree import multipoles_from_particles as cj_multipoles_from_particles, cj_coarsen_multipoles

@pytest.fixture
def posz():
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (1024**2,3))
    posz, isort = cj.tree.pos_zorder_sort(pos0)
    return posz

@pytest.fixture
def particlesz(posz):
    return nt.Particles(posz, jnp.ones(posz.shape[0]))

@pytest.fixture
def tree_hierarchy(particlesz):
    cfg = nt.TreeConfig(coarse_fac=4.0)
    ths : list[nt.TreePlane] = jax.block_until_ready(nt.build_tree_hierarchy.jit(particlesz, cfg))
    return ths

def test_tree_hierarchy(tree_hierarchy : list[nt.TreePlane]):
    for tplane  in tree_hierarchy:
        assert jnp.sum(tplane.npart) == tplane.tot_npart
        lvls = tplane.lvl[:tplane.nnodes]
        assert jnp.all((lvls >= -100 ) & (lvls < 100))

def test_tree_multipoles(particlesz : nt.Particles, tree_hierarchy : list[nt.TreePlane]):
    mp = nt.multipoles_from_particles.jit(tree_hierarchy[0], particlesz, p=2)
    mp2 = cj_multipoles_from_particles(tree_hierarchy[0], particlesz, p=2)
    assert jnp.allclose(tree_hierarchy[0].npart, mp.get(0))
    
    for i in range(mp.values.shape[1]):
        assert jnp.allclose(mp.get(i), mp2.get(i), rtol=1e-3)
    assert jnp.allclose(mp.center(), mp2.center(), rtol=1e-6, equal_nan=True)

    mp_coarse = nt.coarsen_multipoles.jit(mp, tree_hierarchy[1])
    mp_coarse2 = cj_coarsen_multipoles(mp2, tree_hierarchy[1])
    assert jnp.allclose(tree_hierarchy[1].npart, mp_coarse.get(0))
    for i in range(mp.values.shape[1]):
        assert jnp.allclose(mp_coarse.get(i), mp_coarse2.get(i), rtol=1e-3, atol=1e-4)