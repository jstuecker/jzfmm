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
    name="organize_pos", loops=10, only_print_run=only_print_run)

btree = timer.timeit_jit(
    fmdj.octree.get_compressed_binary_tree, morton,
    name="binary_tree", loops=10, only_print_run=only_print_run)