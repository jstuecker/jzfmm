import jax
import jax.numpy as jnp
import fmdj
import numpy as np
import custom_jax as cj
from fmdj.utility import Tee, Timer
import sys
import matplotlib.pyplot as plt

sys.stdout = Tee(sys.stdout, open("logs/interactions.log", "a+"))

print(f"============== Starting Interactions Tests ==============")

def time_interactions(timer, N, p, eps=1e-3):
    cfg = fmdj.Config(tags=("ref",))
    cfg_cj = fmdj.Config(tags=("cj", "ref"))


    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32) * 0.05
    mass0 = jnp.ones(len(pos0), dtype=jnp.float32)
    octree, posz, massz, isortz = fmdj.fmm.build_octree_with_multipoles.jit(pos0+0.5, mass0, p=p)
    err, (ilist, nilist) = fmdj.fmm.build_interaction_list.jit(octree)
    err.throw()
    ilist, iranges = fmdj.fmm.organize_interactions.jit(ilist, nilist, sort=False)
    ilist.block_until_ready()

    print("Interaction Nums:", (iranges[1:] - iranges[:-1]), "percentages:", " ".join(["%.1f" % p for p in (iranges[1:] - iranges[:-1]) * 100.0 / nilist]))

    if p <= 5:
        Loc1 = timer.timeit_jit(fmdj.multipoles.evaluate_ilists_node_node.jit,
            octree.xnode, octree.mp, ilist, iranges[0], iranges[1], p=octree.p, use_cj=False,  eps=eps,
            name = "node_node_jax")
    Loc1b = timer.timeit_jit(fmdj.multipoles.evaluate_ilists_node_node.jit,
        octree.xnode, octree.mp, ilist, iranges[0], iranges[1], p=octree.p, use_cj=True, eps=eps,
        name = "node_node_cj")
    if p <= 5:
        Loc2 = timer.timeit_jit(fmdj.multipoles.evaluate_ilists_leaf_to_node.jit,
            octree.xnode, posz, massz, octree.leaf_particle_bounds, ilist, 
            iranges[1], iranges[2], p=octree.p, use_cj=False, max_leaf_size=octree.max_leaf_size, eps=eps)
        Loc2b = timer.timeit_jit(fmdj.multipoles.evaluate_ilists_leaf_to_node.jit,
            octree.xnode, posz, massz, octree.leaf_particle_bounds, ilist, 
            iranges[1], iranges[2], p=octree.p, use_cj=True, eps=eps, name="leaf2node_cj")
        phi1 = timer.timeit_jit(fmdj.multipoles.ilists_node_to_leaf.jit,
            octree.xnode, octree.mp, posz, octree.leaf_particle_bounds, ilist, 
            iranges[2:4], cfg=cfg)
    phi2 = timer.timeit_jit(fmdj.multipoles.ilist_leaf_to_leaf.jit,
        posz, massz, octree.leaf_particle_bounds, ilist, iranges[3:5], cfg=cfg)
    if p <= 5:
        Loc = timer.timeit_jit(fmdj.multipoles.local_to_local_via_height.jit, octree, Loc1+Loc2)
        phi3 = timer.timeit_jit(fmdj.multipoles.evaluate_local.jit, Loc[octree.node_of_particle], 
                                posz - octree.xnode[octree.node_of_particle])
    
        phi4 = timer.timeit_jit(fmdj.fmm.evaluate_interaction_lists.jit,
            octree, posz, massz, ilist, nilist, sort=False, eps=eps,  name="all together")

timer = Timer(print_compile=False, print_warmup=False, loops=4)
for N in int(1e4), int(1e5), int(3e5), int(1e6), int(2e6):
    timer.set_tag(N=N)

    time_interactions(timer, N, p=4, eps=1e-2)

timer.plot_timings("N", save="logs/interactions_timings_N.pdf")

print("----------- Time p ------------------")

timer = Timer(print_compile=False, print_warmup=False, loops=10)
for p in (1,2,3,4,5):
    timer.set_tag(p=p)

    time_interactions(timer, N=int(2e6), p=p, eps=1e-2)

plt.figure()
timer.plot_timings("p", save="logs/interactions_timings_p.pdf", logx=False)