import fmdj.new_tree as nt
import jax
import jax.numpy as jnp
import custom_jax as cj
import pytest
from fmdj.config import Config, TreeConfig

@pytest.fixture
def posz():
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (1024**2,3))
    posz, isort = cj.tree.pos_zorder_sort(pos0)
    return posz

@pytest.fixture
def particlesz(posz):
    return nt.Particles(posz, jnp.ones(posz.shape[0]))

@pytest.fixture
def tree_hierarchy(particlesz, cfg_cuda):
    ths : list[nt.TreePlane] = jax.block_until_ready(nt.build_tree_hierarchy.jit(particlesz, cfg=cfg_cuda))
    return ths

@pytest.fixture
def cfg_base():
    return Config(tags=("base",))

@pytest.fixture
def cfg_cuda():
    return Config(tags=("cuda", "base"))

def test_tree_hierarchy(tree_hierarchy : list[nt.TreePlane]):
    for tplane  in tree_hierarchy:
        assert jnp.sum(tplane.npart) == tplane.tot_npart
        lvls = tplane.lvl[:tplane.nnodes]
        assert jnp.all((lvls >= -100 ) & (lvls < 100))

def test_tree_multipoles(particlesz: nt.Particles, tree_hierarchy: list[nt.TreePlane], 
                         cfg_base: Config, cfg_cuda: Config):
    mp_base = nt.multipoles_from_particles.jit(tree_hierarchy[0], particlesz, cfg=cfg_base)
    mp_cuda = nt.multipoles_from_particles.jit(tree_hierarchy[0], particlesz, cfg=cfg_cuda)
    assert jnp.allclose(tree_hierarchy[0].npart, mp_base.get(0))
    
    for i in range(mp_base.values.shape[1]):
        assert jnp.allclose(mp_base.get(i), mp_cuda.get(i), rtol=1e-3)
    assert jnp.allclose(mp_base.center(), mp_cuda.center(), rtol=1e-6, equal_nan=True)

    mp_coarse = nt.coarsen_multipoles.jit(mp_base, tree_hierarchy[1], cfg=cfg_base)
    mp_coarse2 = nt.coarsen_multipoles.jit(mp_cuda, tree_hierarchy[1], cfg=cfg_cuda)
    assert jnp.allclose(tree_hierarchy[1].npart, mp_coarse.get(0))
    for i in range(mp_base.values.shape[1]):
        assert jnp.allclose(mp_coarse.get(i), mp_coarse2.get(i), rtol=1e-3, atol=1e-4)