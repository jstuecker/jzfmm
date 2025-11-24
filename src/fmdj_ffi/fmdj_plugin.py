from fmdj.variants import Variant, V, VariantManager, vm, has_gpu
from fmdj import config
import jax.numpy as jnp

TAG = "cuda"

variants = VariantManager()

import fmdj_ffi as cj
import fmdj_ffi.cj_new_tree as cnt

def register():
    if has_gpu():
        variants[V.ilist_node_to_node][TAG] = Variant(cj.multipoles.ilist_node_to_node)
        variants[V.ilist_leaf_to_node][TAG] = Variant(cj.multipoles.ilist_leaf_to_node)
        variants[V.multipoles_from_particles][TAG] = Variant(cnt.multipoles_from_particles)
        variants[V.coarsen_multipoles][TAG] = Variant(cnt.coarsen_multipoles)
        variants[V.evaluate_plane_interactions][TAG] = Variant(cnt.cj_evaluate_tree_plane)

        vm.register_variants(variants)
        print("Custom Jax Plugin registered!")
    else:
        print("Could not find GPU, Custom Jax Plugin not registered!")