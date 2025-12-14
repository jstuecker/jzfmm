import jax.numpy as jnp
import jax
import fmdj
from fmdj.data import PosMass, TreePlane, TreeHierarchy
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

def get_pos(N=5555, xmin=0., xmax=1., seed=1):
    pos0 = jax.random.uniform(
        jax.random.PRNGKey(seed), (N,3), dtype=jnp.float32, minval=xmin, maxval=xmax
    )
    
    return pos0

def test_search_sorted_z():
    posz, idz = fmdj.ztree.pos_zorder_sort.jit(get_pos(1387, xmin=0.1, xmax=0.4, seed=0))
    posz2, idz2 = fmdj.ztree.pos_zorder_sort.jit(get_pos(2222, xmin=0.1, xmax=0.4, seed=2))

    iself = fmdj.ztree.search_sorted_z.jit(posz, posz)
    assert jnp.all(iself == jnp.arange(len(posz), dtype=jnp.int32))

    i2 = fmdj.ztree.search_sorted_z.jit(posz, posz2)
    pos_ins = jnp.insert(posz, i2, posz2, axis=0)
    pos_ins_ref = fmdj.ztree.pos_zorder_sort.jit(pos_ins)[0]
    assert jnp.all(pos_ins == pos_ins_ref), "If indices were right, we should already be in z-order"

def test_leaf_search(pos_mass_z: PosMass, tree_hierarchy: TreeHierarchy):
    # Check whether we can learn the right leaf numbers just from the leaf positions
    nleaves = tree_hierarchy.lvl.num(0)
    xleaf = tree_hierarchy.geom_cent.get(0, nleaves)
    spl = tree_hierarchy.ispl_n2n.get(0, nleaves+1)

    ileaf = fmdj.ztree.search_sorted_z(xleaf, pos_mass_z.pos, leaf_search=True)
    spl2 = jnp.searchsorted(ileaf, jnp.arange(len(xleaf)+1), side="left")

    assert jnp.all(spl == spl2), "Leaf ranges should be identical"