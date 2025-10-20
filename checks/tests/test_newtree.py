import fmdj.new_tree as nt
import jax
import jax.numpy as jnp
import custom_jax as cj
import pytest

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
    assert jnp.allclose(tree_hierarchy[0].npart, mp.get(0))

    mp_coarse = nt.coarsen_multipoles.jit(mp, tree_hierarchy[1])
    assert jnp.allclose(tree_hierarchy[1].npart, mp_coarse.get(0))