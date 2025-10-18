import custom_jax as cj
import pytest
import fmdj
import jax
import jax.numpy as jnp
import numpy as np

# Here we test whether the new custom JAX functions work produce identical results
# to the pure jax functions

def setup_particles(N=5555, duplicate=False):
    pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32, minval=-0.5, maxval=0.5)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))
    mass0 = jnp.ones(len(pos0), dtype=jnp.float32)
    
    return pos0, mass0

def cj_binary_tree(pos0, mass0):
    posz, isort_z = cj.tree.pos_zorder_sort(pos0)
    btree_z = fmdj.octree.cj_build_ztree(posz)
    return btree_z, posz, isort_z

def test_compare_cjtree_self():
    """Test whether our custom CUDA kernels give reproducible results"""
    pos0, mass0 = setup_particles(duplicate=False)
    btree1, posz1, isortz1 = cj_binary_tree(pos0, mass0)
    btree2, posz2, isortz2 = cj_binary_tree(pos0, mass0)

    assert np.all(isortz1 == isortz2)
    assert np.all(posz1 == posz2)

    assert np.all(btree1.level_binary == btree2.level_binary)
    assert np.all(btree1.lbound == btree2.lbound)
    assert np.all(btree1.rbound == btree2.rbound)
    assert np.all(btree1.lchild == btree2.lchild)
    assert np.all(btree1.rchild == btree2.rchild)

def test_compare_octrees():
    pos0, mass0 = setup_particles(duplicate=False)
    octree1, posz1, massz1, isortz1 = fmdj.octree.sort_and_build_octree(pos0+0.5, mass0, use_cj=True)
    octree2, posz2, massz2, isortz2 = fmdj.octree.sort_and_build_octree(pos0, mass0, use_cj=False)

    assert np.all(isortz1 == isortz2)

    # Float tree has a different max level, that's why we cannot compare the first and last level
    assert np.all(octree1.level_binary[1:octree1.nnodes-1] == octree2.level_binary[1:octree2.nnodes-1])

    for key in ("parent", "lchild", "rchild", "is_valid", "height"):
        assert np.all(octree1.__getattribute__(key)[:octree1.nnodes] 
                      == octree2.__getattribute__(key)[:octree2.nnodes])