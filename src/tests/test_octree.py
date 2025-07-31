import pytest
import fmdj
import jax
import jax.numpy as jnp

def setup_particles(N=5555, duplicate=False):
    pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32, minval=-0.5, maxval=0.5)
    if duplicate:
        pos0 = jnp.concatenate((pos0, pos0, pos0, pos0))

    morton, pos, isort = fmdj.octree.organize_particles(pos0)

    mass = jnp.ones(len(pos), dtype=jnp.float32)
    
    return morton, pos, mass

def test_btree_children():
    morton = setup_particles()[0]
    tree = fmdj.octree.get_compressed_binary_tree(morton)

    # Parent of child should be the node itself
    inode = jnp.where(tree.lchild > 0)[0]
    assert jnp.all(inode == tree.rbound[tree.lchild[inode]])

    inode = jnp.where((tree.rchild > 0) & (tree.level_binary >= 0))[0]
    assert jnp.all(inode == tree.lbound[tree.rchild[inode]])

@pytest.mark.parametrize("duplicate", [False, True])
def test_otree_reduction(duplicate):
    """A reduced octree should be equivalent to a tree build on its leaf-nodes"""
    morton, pos, mass = setup_particles(duplicate=duplicate)
    btree = fmdj.octree.get_compressed_binary_tree(morton)
    octree = fmdj.octree.get_reduced_octree(btree, pos, mass, max_leaf_size=3)

    nleaves = octree.nnodes-1
    xleaf = octree.xleaf[:nleaves]
    morton_leaf = fmdj.octree.organize_particles(xleaf)[0]
    btree2 = fmdj.octree.get_compressed_binary_tree(morton_leaf)

    assert jnp.all(octree.lchild[:octree.nnodes] == btree2.lchild)
    assert jnp.all(octree.rchild[:octree.nnodes] == btree2.rchild)
    assert jnp.all(octree.level_binary[:octree.nnodes] == btree2.level_binary)

    # Nodesize should be meaningful
    nodesize = octree.leaf_particle_bounds[1:nleaves] - octree.leaf_particle_bounds[:nleaves-1]
    assert jnp.all(nodesize >= 1)

    if not duplicate:
        assert jnp.all(nodesize <= octree.max_leaf_size)

    # All leaf positions should be valid
    assert jnp.all(~jnp.isnan(octree.xleaf[:octree.nnodes-1]))

