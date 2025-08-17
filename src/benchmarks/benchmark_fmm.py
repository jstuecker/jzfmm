import sys
import fmdj
from fmdj.utility import Tee, Timer
import jax.numpy as jnp
import jax

sys.stdout = Tee(sys.stdout, open("logs/fmm.log", "a+"))

N = 1024*1024
pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
mass = jnp.ones(N, dtype=jnp.float32)

print(f"============== Starting FMM Tests (N={N:.1e})  ==============")
timer = Timer(verbose=True, print_compile=False, print_warmup=False)

octree, pos, mass, isort = timer.timeit_jit(
    fmdj.fmm.build_octree_with_multipoles.jit, pos0, mass, max_leaf_size=64, p=3)

for thetamax in (0.75, 1.0):
    timer.set_tag(opening_angle=thetamax)
    err, (ilist, nilist) = timer.timeit_jit(fmdj.fmm.build_interaction_list.jit,
        octree, thetamax=thetamax, ilist_fac=512, name="build_ilist")
    print(f"Ilist size: {nilist:.1e}/{len(ilist):.1e} for thetamax={thetamax}")
    err.throw()

    ilist, isplits = timer.timeit_jit(fmdj.fmm.organize_interactions.jit, ilist, nilist, 
                                      sort=False, name="organize_interactions_nosort")

timer.plot_timings("opening_angle", save="logs/fmm_timings.pdf", logx=False)