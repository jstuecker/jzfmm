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
    return octree, pos

only_print_run = True
print(f"============== Starting Multipole Tests (N={N:.1e})  ==============")

loops = 40

timer = Timer(verbose=True)

octree, pos = timer.timeit_jit(
    sort_and_build_tree, pos0, mass, name="sort_and_build_tree", 
    loops=loops, only_print_run=only_print_run)

m, xcom = timer.timeit_jit(
    fmdj.multipoles.com_via_levels, octree, pos, mass,
    name="com_via_levels", only_print_run=only_print_run, loops=loops)

for p in (2,3,4,5):
    mp = timer.timeit_jit(
        fmdj.multipoles.multipoles_via_levels, octree, pos, mass,
        static_argnames=("p",), p=p, xcom=xcom,
        name="mp_via_levels_p%d"%p, only_print_run=only_print_run, loops=loops)

print("------------- Combined:  ")

octree, pos_sorted, mass_sorted, isort = timer.timeit_jit(
    fmdj.fmm.build_octree_with_multipoles, pos0, mass,
    static_argnames=("max_leaf_size", "p"), max_leaf_size=64, p=3,
    name="build_octree_with_multipoles_p3", only_print_run=only_print_run, loops=loops)