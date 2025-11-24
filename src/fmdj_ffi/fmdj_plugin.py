from fmdj.variants import Variant, V, VariantManager, vm, has_gpu
from fmdj import config
import jax.numpy as jnp

TAG = "cuda"

variants = VariantManager()

import fmdj_ffi as cj
import fmdj_ffi.cj_new_tree as cnt

def register():
    if has_gpu():

        vm.register_variants(variants)
        print("Custom Jax Plugin registered!")
    else:
        print("Could not find GPU, Custom Jax Plugin not registered!")