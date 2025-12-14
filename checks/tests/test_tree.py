import jax.numpy as jnp
import jax
import fmdj
from fmdj.data import PosMass, TreePlane
import fmdj_jaxonly.jaxonly_multipoles as jmp
import pytest
import numpy.testing as npt

def test_tree_hierarchy(tree_planes : list[TreePlane]):
    npart = jnp.sum(tree_planes[0].npart)
    for tplane  in tree_planes:
        assert jnp.sum(tplane.npart) == npart
        lvls = tplane.lvl[:tplane.nnodes]
        assert jnp.all((lvls >= -100 ) & (lvls < 100))

def test_node_geometry(pos_mass_z: PosMass):
    pos = pos_mass_z.pos
    pos = pos[::500]
    ispl = jnp.sort(jax.random.randint(jax.random.PRNGKey(42), (50,), 0, pos.shape[0]))

    lvl, cent, ext = fmdj.ztree.get_node_geometry(pos, ispl[:-1], ispl[1:])

    for i in range(len(ispl)-1):
        if ispl[i+1] - ispl[i] <= 1:
            continue
        pnode = pos[ispl[i]:ispl[i+1]]

        # print("----")
        # print(pnode[0], pnode[-1])
        # print(jnp.min(pnode, axis=0), cent[i] - ext[i]*0.5)
        # print(jnp.mean(pnode, axis=0), cent[i])
        # print(jnp.max(pnode, axis=0), cent[i] + ext[i]*0.5)

        pmin, pmax = jnp.min(pnode, axis=0), jnp.max(pnode, axis=0)
                
        npt.assert_array_less(cent[i] - ext[i]*0.5, pmin)
        npt.assert_array_less(pmax, cent[i] + ext[i]*0.5)

        # Check that node is not too big, by seeing that half the extent would not be sufficient
        if jnp.all(jnp.isfinite(ext[i])):
            assert jnp.any(cent[i] - ext[i]*0.25 >= pmin) or jnp.any(cent[i] + ext[i]*0.25 <= pmax)