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

def test_tree_hierarchy(posz):
    cfg = nt.TreeConfig(alloc_fac_nodes=1.2, coarse_fac=4.0)
    ths = jax.block_until_ready(nt.build_level_hierarchy.jit(posz, cfg))

    for th in ths:
        assert jnp.sum(th.npart) == th.num_part
        lvls = th.lvl[:th.nnodes]
        assert jnp.all((lvls >= -100 ) & (lvls < 100))