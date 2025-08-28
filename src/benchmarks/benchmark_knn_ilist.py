import jax
import jax.numpy as jnp
import fmdj
import custom_jax as cj
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fmdj.utility import Tee, Timer
import sys
import numpy as np

boxsize = 1.
k = 16
N = 512*1024

sys.stdout = Tee(sys.stdout, open("logs/knn_ilist.log", "a+"))

print(f"============== Ilist knn tests ==============")
timer = Timer(verbose=True, loops=50, print_compile=False, print_warmup=False)
timer.set_tag(N=N, k=k)

pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), minval=0, maxval=1, dtype=jnp.float32)

octree, posz, massz, isort_z = fmdj.fmm.build_octree_with_multipoles.jit(
    pos0, jnp.ones(pos0.shape[0]), use_cj=True, max_leaf_size=32, p=1)

res = cj.knn.brute_force_node_ilist_prep(octree, k=16)
leaf_cent, level_leaf = res[0], res[1]

radii, il, ilr, ispl = cj.knn.build_ilist_knn.jit(
    *res, alloc_fac=180, k=k, sort=True, boxsize=boxsize, sort_alloc_fac=2.2)

rnn, inn = cj.knn.ilist_knn_search.jit(posz, octree.leaf_particle_bounds, leaf_cent, level_leaf, 
                                       il, ispl, k=k, boxsize=boxsize)

tree = cKDTree(posz, boxsize=boxsize if boxsize > 0 else None)
rknn2, iknn2 = tree.query(posz, k=k)
print(jnp.allclose(rnn, rknn2))
print(jnp.allclose(inn, iknn2), jnp.sum(inn != iknn2))

timer.timeit_jit(cj.knn.build_ilist_knn.jit, *res, alloc_fac=180, k=k, sort=True, name="sort")
timer.timeit_jit(cj.knn.build_ilist_knn.jit, *res, alloc_fac=180, k=k, sort=False, name="nosort")