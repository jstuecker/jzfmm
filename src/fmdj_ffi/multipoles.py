from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from fmdj_cuda import ffi_multipoles as ffi_multipoles

jax.ffi.register_ffi_target("IlistM2L", ffi_multipoles.IlistM2L(), platform="CUDA")

# ======= Multipole Translators =======

def ilist_node_to_node(xnodes, multipoles, interactions, irange=None, block_size=32, interactions_per_block=None, cfg=None):
    p = cfg.fmm.p
    softening = cfg.softening

    assert xnodes.dtype == jnp.float32
    assert multipoles.dtype == jnp.float32
    assert xnodes.ndim >= 2 and multipoles.ndim >= 2 and multipoles.ndim >= 2
    assert xnodes.shape[-1] == 3
    assert multipoles.shape[-1] == ((p+3)*(p+2)*(p+1)) // 6

    if interactions is None:
        iarange = jnp.arange(0, len(multipoles))
        interactions = np.stack((iarange, iarange), axis=-1).astype(np.int32)
    if irange is None:
        irange = jnp.array([0, len(interactions)], dtype=jnp.int32)
    if interactions_per_block is None:
        interactions_per_block = np.clip(len(interactions) // (8096*block_size), 1, 256)

    out_type = jax.ShapeDtypeStruct(multipoles.shape, multipoles.dtype)
    loc = jax.ffi.ffi_call("IlistM2L", (out_type,))(xnodes, multipoles, interactions, irange, p=np.int32(p), block_size=np.uint64(block_size), interactions_per_block=np.uint64(interactions_per_block), epsilon=np.float32(softening))[0]
    
    return loc
ilist_node_to_node.jit = jax.jit(ilist_node_to_node, static_argnames=("block_size", "cfg"))