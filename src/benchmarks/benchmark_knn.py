import jax
import jax.numpy as jnp
import fmdj
import custom_jax as cj
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fmdj.utility import Tee, Timer
import sys
import numpy as np

sys.stdout = Tee(sys.stdout, open("logs/knn.log", "a+"))

print(f"============== Starting KNN Tests  ==============")

timer = Timer(verbose=True, loops=40, print_compile=False, print_warmup=False)

def setup(N=1024*1024, k=16):
    pos0 = jax.random.uniform(jax.random.PRNGKey(0), (N,3), minval=0, maxval=1, dtype=jnp.float32)

    octree, posz, massz, isort_z = fmdj.fmm.build_octree_with_multipoles.jit(pos0, jnp.ones(pos0.shape[0]), use_cj=True, max_leaf_size=32, p=1)

    npart_leaf = octree.leaf_particle_bounds[1:] - octree.leaf_particle_bounds[:-1]
    level_leaf = octree.level_binary[octree.node_of_leaf] - 1

    leaf_cent, leaf_ext = cj.knn.get_node_box(octree.xleaf, level_leaf)
    rmin, ilist, isplit = cj.knn.knn_interactions.jit(leaf_cent, leaf_ext, npart_leaf, alloc_fac=128, k=32)
    assert isplit[-1] < len(ilist)

    nleaves = octree.nnodes - 1
    leaf_level = octree.level_binary[octree.node_of_leaf] - 1
    leaf_isplit = octree.leaf_particle_bounds[:nleaves+1]
    leaf_level = leaf_level[:nleaves]
    leaf_cent = leaf_cent[:nleaves]

    return posz, leaf_isplit, leaf_cent, leaf_level, ilist, isplit

def run_or_load(N = 1024*1024, k = 16, redo=False):
    filename = f"logs/tmp/knn_{k}_{N}.npz"

    if redo:
        # print("Running setup for N =", N, " k =", k)
        res = setup(N=1024*1024, k=k)
        np.savez(filename, *res)
    else:
        # print("Loading from ", filename)
        res = np.load(filename)
        res = [jnp.array(res[k]) for k in res]
    return res

k = 16
posz, leaf_isplit, leaf_cent, leaf_level, ilist, isplit = run_or_load(N=1024*1024, k=k, redo=False)

rnn, inn = cj.knn.ilist_knn_search.jit(posz, leaf_isplit, leaf_cent, leaf_level, ilist, isplit, k=k, interactions_per_block=1)

tree = cKDTree(posz)
rknn2, iknn2 = tree.query(posz, k=k)

print(jnp.allclose(rnn, rknn2))
print(jnp.allclose(inn, iknn2), jnp.sum(inn != iknn2))

for k in (4,8,16,32):
    timer.set_tag(N=1024**2, k=k)
    posz, leaf_isplit, leaf_cent, leaf_level, ilist, isplit = run_or_load(N=1024*1024, k=k, redo=False)
    rnn, inn = timer.timeit_jit(cj.knn.ilist_knn_search.jit, posz, leaf_isplit, leaf_cent, leaf_level, ilist, isplit, k=k)