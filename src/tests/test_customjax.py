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

def cj_sort_and_build_tree(pos0, mass0):
    posz, isort_z = cj.tree.pos_zorder_sort(pos0)
    btree_z = fmdj.octree.cj_build_ztree(posz)
    octree = fmdj.octree.get_reduced_octree(btree_z, posz, mass0[isort_z], max_leaf_size=64)
    return btree_z, octree, posz, isort_z

def sort_and_build_tree(pos0, mass):
    morton, pos, isort = fmdj.octree.organize_particles(pos0)
    btree = fmdj.octree.get_compressed_binary_tree(morton)
    octree = fmdj.octree.get_reduced_octree(btree, pos, mass, max_leaf_size=64)
    return octree, isort

def test_compare_cjtree_self():
    """Test whether our custom CUDA kernels give reproducible results"""
    pos0, mass0 = setup_particles(duplicate=False)
    btree1, octree1, posz1, isortz1 = cj_sort_and_build_tree(pos0, mass0)
    btree2, octree2, posz2, isortz2 = cj_sort_and_build_tree(pos0, mass0)

    assert np.all(isortz1 == isortz2)
    assert np.all(posz1 == posz2)

    assert np.all(btree1.level_binary == btree2.level_binary)
    assert np.all(btree1.lbound == btree2.lbound)
    assert np.all(btree1.rbound == btree2.rbound)
    assert np.all(btree1.lchild == btree2.lchild)
    assert np.all(btree1.rchild == btree2.rchild)

def test_compare_octrees():
    pos0, mass0 = setup_particles(duplicate=False)
    btree1, octree1, posz1, isortz1 = cj_sort_and_build_tree(pos0+0.5, mass0)
    octree2, isort2 = sort_and_build_tree(pos0, mass0)

    assert np.all(isortz1 == isort2)

    # Float tree has a different max level, that's why we cannot compare the first and last level
    assert np.all(octree1.level_binary[1:octree1.nnodes-1] == octree2.level_binary[1:octree2.nnodes-1])

    for key in ("parent", "lchild", "rchild", "is_valid", "height"):
        assert np.all(octree1.__getattribute__(key)[:octree1.nnodes] 
                      == octree2.__getattribute__(key)[:octree2.nnodes])