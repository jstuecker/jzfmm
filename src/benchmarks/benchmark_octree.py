import sys
import fmdj
from fmdj.utility import Tee, Timer
import jax.numpy as jnp
import jax
try:
    import custom_jax as cj
except ImportError:
    print("No custom JAX found... skipping some tests.")
    cj = None

sys.stdout = Tee(sys.stdout, open("logs/octree.log", "a+"))

N = 1024*1024*8
print(f"============== Starting Octree Tests (N={N:.1e})  ==============")

def create_pos():
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
    return jnp.clip(pos0, -0.5, 0.5)

timer = Timer(verbose=True, print_compile=False, print_warmup=False)
pos0 = timer.timeit_jit(create_pos, name="create_pos")
mass0 = jnp.ones(N, dtype=jnp.float32)

morton, pos, isort = timer.timeit_jit(fmdj.octree.organize_particles.jit, pos0, return_sorted=True)

morton_unsorted = fmdj.octree.organize_particles(pos0, return_sorted=False)[0].block_until_ready()
timer.timeit_jit(jnp.lexsort, morton_unsorted.T)
timer.timeit_jit(jnp.argsort, morton_unsorted[...,2], stable=False,  name="argsort_i32")

levels = timer.timeit_jit(fmdj.octree.morton_diff_level, morton[1:], morton[:-1])

lbound, rbound = timer.timeit_jit(fmdj.octree.find_previous_and_next_higher.jit, levels)

lbound, rbound = timer.timeit_jit(fmdj.octree.determine_children.jit, levels, lbound, rbound)

btree = timer.timeit_jit(fmdj.octree.get_compressed_binary_tree.jit, morton)

octree = timer.timeit_jit(fmdj.octree.get_reduced_octree.jit, btree, pos, mass0, max_leaf_size=64)

octree2 = timer.timeit_jit(fmdj.octree.put_nodes_in_level_order.jit, octree)

timer.timeit_jit(fmdj.octree.get_tree_height.jit, btree, name="get_tree_height_btree")
timer.timeit_jit(fmdj.octree.get_tree_height.jit, octree, name="get_tree_height_otree64")

if cj is not None:
    posz, isort_z = timer.timeit_jit(cj.tree.pos_zorder_sort.jit, pos, name="cj_zsort")
    btree_z = timer.timeit_jit(fmdj.octree.cj_build_ztree.jit, posz)

print("-------------")

@jax.jit
def sort_and_build_tree(pos0, mass):
    morton, pos = fmdj.octree.organize_particles(pos0)[0:2]
    btree = fmdj.octree.get_compressed_binary_tree(morton)
    octree = fmdj.octree.get_reduced_octree(btree, pos, mass, max_leaf_size=64)
    return octree

octree = timer.timeit_jit(sort_and_build_tree, pos0, mass0, loops=100)

@jax.jit
def cj_sort_and_build_tree(pos0, mass0):
    posz, isort_z = cj.tree.pos_zorder_sort(pos0)
    btree_z = fmdj.octree.cj_build_ztree(posz)
    octree = fmdj.octree.get_reduced_octree(btree_z, posz, mass0[isort_z], max_leaf_size=64)
    return octree
octree = timer.timeit_jit(cj_sort_and_build_tree, pos0, mass0, loops=100)