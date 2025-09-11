import jax
import jax.numpy as jnp
import custom_jax as cj
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fmdj.utility import Tee, Timer
import sys
import numpy as np

boxsize = 0.
k = 16
N = 512*1024

sys.stdout = Tee(sys.stdout, open("logs/knn_ilist.log", "a+"))

print(f"============== Lowest level Ilist build (alloc 300) ==============")
timer = Timer(verbose=True, loops=50, print_compile=False, print_warmup=False)
timer.set_tag(N=N, k=k)

pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), minval=0, maxval=1, dtype=jnp.float32)
posz, idz = cj.tree.pos_zorder_sort.jit(pos0)


spl, nleaf, llvl, xleaf, numleaves = cj.tree.summarize_leaves.jit(posz, max_size=32)
spl2, nleaf2, llvl2, xleaf2, numleaves2 = cj.tree.summarize_leaves.jit(
    xleaf, max_size=32*8, nleaf=nleaf, num_part=len(posz))
il, ispl = cj.knn.build_ilist_recursive.jit(
    xleaf2, llvl2, nleaf2, max_size=32*64, refine_fac=8, num_part=len(posz), k=16, boxsize=boxsize)

par = (xleaf, llvl, nleaf, spl2, il, ispl)

radii, il, ispl = cj.knn.build_ilist_knn.jit(*par, alloc_fac=180, k=k, sort=False, boxsize=boxsize)

rnn, inn = cj.knn.ilist_knn_search.jit(posz, spl, xleaf, llvl, il, ispl, k=k, boxsize=boxsize)

tree = cKDTree(posz, boxsize=boxsize if boxsize > 0 else None)
rknn2, iknn2 = tree.query(posz, k=k)
print(jnp.allclose(rnn, rknn2))
print(jnp.allclose(inn, iknn2), jnp.sum(inn != iknn2))

for rfac in 2, 4, 8, 16:
    timer.set_tag(rfac=rfac)
    spl2, nleaf2, llvl2, xleaf2, numleaves2 = cj.tree.summarize_leaves.jit(
        xleaf, max_size=32*rfac, nleaf=nleaf, num_part=len(posz))
    il, ispl = cj.knn.build_ilist_recursive.jit(
        xleaf2, llvl2, nleaf2, max_size=32*rfac*rfac, refine_fac=8, num_part=len(posz), k=16, boxsize=boxsize)
    par = (xleaf, llvl, nleaf, spl2, il, ispl)
    timer.timeit_jit(cj.knn.build_ilist_knn.jit, *par, alloc_fac=300, k=k, sort=False, name="nosort")
    # timer.timeit_jit(cj.knn.build_ilist_knn.jit, *par, alloc_fac=180, k=k, sort=True, name="sort")