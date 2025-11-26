import jax.numpy as jnp
import fmdj
from fmdj.data import PosMass, TreePlane
import fmdj_jaxonly.jaxonly_multipoles as jmp

def test_tree_hierarchy(tree_hierarchy : list[TreePlane]):
    for tplane  in tree_hierarchy:
        assert jnp.sum(tplane.npart) == tplane.tot_npart
        lvls = tplane.lvl[:tplane.nnodes]
        assert jnp.all((lvls >= -100 ) & (lvls < 100))