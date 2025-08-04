import sys
import fmdj
from fmdj.utility import Tee, Timer
import jax.numpy as jnp
import jax

sys.stdout = Tee(sys.stdout, open("logs/multipoles.log", "a+"))


print(f"============== Starting Multipole Tests  ==============")

timer = Timer(verbose=True, loops=40, print_compile=False, print_warmup=False)

for N in (int(1e4), int(1e5), int(1e6), int(3e6), int(1e7)):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
    pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
    mass = jnp.ones(N, dtype=jnp.float32).block_until_ready()

    timer.set_tag(N=N)

    octree, posz, mass0, isortz = timer.timeit_jit(fmdj.octree.sort_and_build_octree.jit, pos0, mass, use_cj=True)

    m, xcom = timer.timeit_jit(fmdj.multipoles.com_via_height.jit, octree, posz, mass)

    for p in (2,3,4,5):
        mp = timer.timeit_jit(fmdj.multipoles.multipoles_via_height.jit, 
                            octree, posz, mass, p=p, xcom=xcom, name="mp_via_height_p%d"%p)

    print("------------- Combined:  ")

    octree, pos_sorted, mass_sorted, isort = timer.timeit_jit(
        fmdj.fmm.build_octree_with_multipoles.jit, pos0, mass, max_leaf_size=64, p=3, use_cj=True, 
        name="build_octree_with_multipoles_p3")
    
timer.plot_timings("N", save="logs/multipoles_timings.pdf")