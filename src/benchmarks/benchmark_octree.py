import sys
import fmdj
from fmdj.utility import Tee, Timer
import jax.numpy as jnp
import jax

sys.stdout = Tee(sys.stdout, open("logs/octree.log", "a+"))

N = 1024*1024
only_print_run = True
print(f"============== Starting Octree Tests (N={N:.1e})  ==============")

def create_pos():
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
    return jnp.clip(pos0, -0.5, 0.5)

timer = Timer(verbose=True)
pos0 = timer.timeit_jit(create_pos, name="create_pos", loops=10, only_print_run=only_print_run)

morton, pos, isort = timer.timeit_jit(
    fmdj.octree.organize_particles, pos0, static_argnames=("return_sorted"), 
    name="organize_particles", loops=10, only_print_run=only_print_run)

# btree = timer.timeit_jit(
#     fmdj.octree.get_compressed_binary_tree, morton,
#     name="binary_tree", loops=10, only_print_run=only_print_run)

levels = timer.timeit_jit(
    fmdj.octree.morton_diff_level, morton[1:], morton[:-1],
    name="morton_diff_level", loops=10, only_print_run=only_print_run)

lbound, rbound = timer.timeit_jit(
    fmdj.octree.find_previous_and_next_lower, levels,
    name="find_previous_and_next_lower", loops=10, only_print_run=only_print_run)

lbound, rbound = timer.timeit_jit(
    fmdj.octree.determine_children, levels, lbound, rbound,
    name="determine_children", loops=10, only_print_run=only_print_run)

btree = timer.timeit_jit(
    fmdj.octree.get_compressed_binary_tree, morton,
    name="get_compressed_binary_tree", loops=10, only_print_run=only_print_run)

print("-------------")



# btree = fmdj.octree.get_compressed_binary_tree(morton, version=1)
# print(btree.lbound[0:10])
# print(btree.lchild[0:10])
# btree = fmdj.octree.get_compressed_binary_tree(morton, version=2)
# print(btree.lbound[0:10])
# print(btree.lchild[0:10])