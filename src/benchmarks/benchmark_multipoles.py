import sys
import fmdj
from fmdj.utility import Tee, Timer
import jax.numpy as jnp
import jax

sys.stdout = Tee(sys.stdout, open("logs/multipoles.log", "a+"))

def sort_and_build_tree(pos0, mass):
    morton, pos = fmdj.octree.organize_particles(pos0)[0:2]
    btree = fmdj.octree.get_compressed_binary_tree(morton)
    octree = fmdj.octree.get_reduced_octree(btree, pos, mass, max_leaf_size=64)
    return octree, pos

print(f"============== Starting Multipole Tests  ==============")

timer = Timer(verbose=True, loops=10, print_compile=False, print_warmup=False)

for N in (int(1e4), int(1e5), int(1e6), int(3e6), int(1e7)):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
    pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
    mass = jnp.ones(N, dtype=jnp.float32).block_until_ready()

    timer.set_tag(N=N)

    octree, pos = timer.timeit_jit(sort_and_build_tree, pos0, mass)

    m, xcom = timer.timeit_jit(fmdj.multipoles.com_via_levels, octree, pos, mass)
    m2, xcom2 = timer.timeit_jit(fmdj.multipoles.com_via_height, octree, pos, mass)

    for p in (2,3,4,5):
        mp = timer.timeit_jit(fmdj.multipoles.multipoles_via_levels.jit, 
                            octree, pos, mass, p=p, xcom=xcom, name="mp_via_levels_p%d"%p)
        mp2 = timer.timeit_jit(fmdj.multipoles.multipoles_via_height.jit, 
                            octree, pos, mass, p=p, xcom=xcom2, name="mp_via_height_p%d"%p)


    print("------------- Combined:  ")

    octree, pos_sorted, mass_sorted, isort = timer.timeit_jit(
        fmdj.fmm.build_octree_with_multipoles.jit, pos0, mass, max_leaf_size=64, p=3, 
        name="build_octree_with_multipoles_p3")
    
timer.plot_timings("N", save="logs/multipoles_timings.pdf")