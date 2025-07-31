import sys
import fmdj
from fmdj.utility import Tee, Timer
import jax.numpy as jnp
import jax

sys.stdout = Tee(sys.stdout, open("logs/multipoles.log", "a+"))

N = 1024*1024
pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
mass = jnp.ones(N, dtype=jnp.float32)

def sort_and_build_tree(pos0, mass):
    morton, pos = fmdj.octree.organize_particles(pos0)[0:2]
    btree = fmdj.octree.get_compressed_binary_tree(morton)
    octree = fmdj.octree.get_reduced_octree(btree, pos, mass, max_leaf_size=64)
    return octree

only_print_run = False
print(f"============== Starting Multipole Tests (N={N:.1e})  ==============")

timer = Timer(verbose=True)

octree = timer.timeit_jit(
    sort_and_build_tree, pos0, mass, name="sort_and_build_tree", 
    loops=100, only_print_run=only_print_run)

m, xcom = timer.timeit_jit(
    fmdj.multipoles.com_via_levels, octree, pos0, mass,
    name="com_via_levels", only_print_run=only_print_run)

for p in (2,3,4,5):
    mp = timer.timeit_jit(
        fmdj.multipoles.multipoles_via_levels, octree, pos0, mass,
        static_argnames=("p",), p=p, xcom=xcom,
        name="mp_via_levels_p%d"%p, only_print_run=only_print_run)
    