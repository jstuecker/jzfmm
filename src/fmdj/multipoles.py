import jax
import jax.numpy as jnp
import math
import numpy as np
from jztree.data import PackedArray, TreeHierarchy
from .config import Config

import fmdj_cuda.ffi_multipoles as ffi_multipoles
jax.ffi.register_ffi_target("TranslateLocalToLocal", ffi_multipoles.TranslateLocalToLocal(), platform="CUDA")
jax.ffi.register_ffi_target("SummarizeMultipoles", ffi_multipoles.SummarizeMultipoles(), platform="CUDA")
jax.ffi.register_ffi_target("TranslateLocalToLocal_XVJP", ffi_multipoles.TranslateLocalToLocal_XVJP(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                        Some Combinatorics                                        #
# ------------------------------------------------------------------------------------------------ #

def num_multi(p, dim=3):
    if dim == 2:
        return (p + 1) * (p + 2) // 2
    if dim == 3:
        return (p + 1) * (p + 2) * (p + 3) // 6
    return math.comb(p + dim, dim)

def p_of_num_multi(ncomp, dim=3):
    return [num_multi(p, dim=dim) for p in range(20)].index(ncomp)

def iter_multi(p, dim=3, istart=0):
    i = 0

    def iter_fixed_sum(total, ndim):
        if ndim == 1:
            yield (total,)
        else:
            for kd in range(total + 1):
                for rest in iter_fixed_sum(total - kd, ndim - 1):
                    yield rest + (kd,)

    for n in range(p + 1):
        for multi in iter_fixed_sum(n, dim):
            if i >= istart:
                yield multi
            i += 1

def get_index_map(p, dim=3):
    return {c: i for i,c in enumerate(iter_multi(p, dim=dim))}

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def _summarize_multipoles_impl(ispl, mp, xnode, xchild, *, cfg, block_size=32):
    """Summarizes multipoles from child nodes to parent nodes"""
    if len(mp.shape) == 1: # probably plain masses corresponding to monopoles
        mp = mp.reshape(-1,1)

    assert ispl.dtype == jnp.int32
    dtype = mp.dtype
    dim = xnode.shape[-1]
    p_in = p_of_num_multi(mp.shape[1], dim=dim)
    out_mp = jax.ShapeDtypeStruct((ispl.size-1, num_multi(cfg.fmm.p, dim=dim)), dtype)
    mpnew = jax.ffi.ffi_call("SummarizeMultipoles", (out_mp,))(
        ispl, mp, xnode, xchild,
        p_in=np.int32(p_in),
        p=np.int32(cfg.fmm.p),
        block_size=np.uint64(block_size),
        kahan = cfg.fmm.kahan_summation
    )[0]

    return mpnew

def summarize_multipoles(
        ispl: jax.Array,
        mp: jax.Array,
        xnode: jax.Array,
        xchild: jax.Array,
        *, cfg: Config
    ) -> jax.Array:

    if mp.ndim == 1:
        mp = mp.reshape(-1,1)
    dim = xnode.shape[-1]
    pin = p_of_num_multi(mp.shape[-1], dim=dim)

    @jax.custom_vjp
    def eval(xchild, mp):
        return _summarize_multipoles_impl(ispl, mp, xnode, xchild, cfg=cfg)
    
    def eval_fwd(xchild, mp):
        mp_n = _summarize_multipoles_impl(ispl, mp, xnode, xchild, cfg=cfg)
        return mp_n, (mp, xchild)
    
    def eval_bwd(res, gmp_n):
        mp, xchild = res
        gmp = _fmm_node_to_child(ispl, gmp_n, xnode, xchild, cfg=cfg, pout=pin)
        gx = shift_local_to_children_vjp_x(ispl, gmp_n, xnode, xchild, mp)

        return gx, gmp
    
    eval.defvjp(eval_fwd, eval_bwd)
    
    return eval(xchild, mp)
summarize_multipoles.jit = jax.jit(summarize_multipoles, static_argnames=['cfg', ])

def build_multipole_hierarchy(th: TreeHierarchy, pos: jax.Array, mp: jax.Array, *, cfg: Config
                              ) -> PackedArray:
    if jnp.ndim(mp) == 0:
        mp = jnp.broadcast_to(mp, pos.shape[:-1] + (1,))
    elif jnp.ndim(mp) == 1:
        if len(mp) == len(pos): # Have shape (N,) for monopole masses
            mp = jnp.reshape(mp, mp.shape + (1,))
        else: # Have shape (p,) and need to broadcast to (N,p)
            mp = jnp.broadcast_to(mp, jnp.broadcast_shapes(pos.shape[:-1] + (1,), jnp.shape(mp)))

    dim = pos.shape[-1]
    size = th.ispl_n2n.size()-1
    mp0 = summarize_multipoles(th.splits_leaf_to_part(size=size+1), mp, th.center().get(0, size), pos, cfg=cfg)

    mph = PackedArray.create_empty(
        (size, num_multi(cfg.fmm.p, dim=dim)), levels=th.num_planes(), dtype=mp0.dtype, fill_values=jnp.nan
    )
    mph = mph.set(0, mp0, th.num(0))

    def handle_level(i, mph):
        mp_coarse = summarize_multipoles(
            ispl=th.ispl_n2n.get(i, size),
            mp=mph.get(i-1, size),
            xnode=th.center().get(i, size),
            xchild=th.center().get(i-1, size),
            cfg=cfg
        )
        return mph.set(i, mp_coarse, th.num(i))
    mph = jax.lax.fori_loop(1, th.num_planes(), handle_level, mph)
    
    return mph
build_multipole_hierarchy.jit = jax.jit(build_multipole_hierarchy, static_argnames=['cfg'])

def _shift_local_to_children_impl(
        ispl: jnp.array,
        loc: jnp.array,
        xnode: jnp.array,
        xchild: jnp.array,
        pout=None,
        block_size=32
    ) -> jnp.array:
    """Shifts local expansions to child nodes"""
    dtype = loc.dtype

    dim = xnode.shape[-1]
    p = p_of_num_multi(loc.shape[1], dim=dim)

    if pout is None:
        pout = p

    assert (pout >= 0) and (pout <= p)

    out_loc = jax.ShapeDtypeStruct((xchild.shape[0], num_multi(pout, dim=dim)), dtype)

    locnew = jax.ffi.ffi_call("TranslateLocalToLocal", (out_loc,))(
        ispl, loc, xnode, xchild,
        p=np.int32(p), pout=np.int32(pout), block_size=np.uint64(block_size)
    )[0]
    return locnew

def shift_local_to_children_vjp_x(
        ispl: jnp.array,
        loc: jnp.array,
        xnode: jnp.array,
        xchild: jnp.array,
        gloc_child: jnp.array,
        block_size=32
    ) -> jnp.array:
    """Shifts local expansions to child nodes"""
    dtype = loc.dtype

    dim = xnode.shape[-1]
    pout = p_of_num_multi(gloc_child.shape[1], dim=dim)
    p = p_of_num_multi(loc.shape[1], dim=dim)

    assert p >= pout
    assert len(ispl) == len(loc) + 1 == len(xnode) + 1

    out_loc = jax.ShapeDtypeStruct(xchild.shape, dtype)

    locnew = jax.ffi.ffi_call("TranslateLocalToLocal_XVJP", (out_loc,))(
        ispl, loc, xnode, xchild, gloc_child,
        p=np.int32(p), pout=np.int32(pout), block_size=np.uint64(block_size)
    )[0]
    return locnew

def _fmm_node_to_child(
        ispl: jnp.array,
        loc: jnp.array,
        xnode: jnp.array,
        xchild: jnp.array,
        cfg: Config,
        pout: int = None
    ) -> jax.Array:
    assert len(ispl) == len(loc) + 1 == len(xnode) + 1

    if loc.ndim == 1:
        loc = loc.reshape(-1,1)
    dim = xnode.shape[-1]
    p = p_of_num_multi(loc.shape[-1], dim=dim)
    if pout is None:
        pout = p_of_num_multi(loc.shape[-1], dim=dim)

    @jax.custom_vjp
    def eval(xchild, loc):
        return _shift_local_to_children_impl(ispl, loc, xnode, xchild, pout=min(pout,p))
    
    def eval_fwd(xchild, loc): # save higher order local expansion for the backward pass
        loc_c = _shift_local_to_children_impl(ispl, loc, xnode, xchild, pout=min(pout,p))
        return loc_c, (loc, xchild)
    
    def eval_bwd(res, gloc_c): # the adjoint of the l2l operator is an m2m operator
        loc, xchild = res
        gloc = summarize_multipoles(ispl, gloc_c, xnode, xchild, cfg=cfg)
        gx = shift_local_to_children_vjp_x(ispl, loc, xnode, xchild, gloc_c)
        return gx, gloc
    
    eval.defvjp(eval_fwd, eval_bwd)
    
    return eval(xchild, loc)
_fmm_node_to_child.jit = jax.jit(_fmm_node_to_child, static_argnames=["cfg", "pout"])

def multipole_readout_pos_vjp(mp: jax.Array, gmp: jax.Array, dim=3):
    return local_readout_pos_vjp(gmp, mp, dim=dim) # turns out, math is identical with transposed inputs

def local_readout_pos_vjp(loc: jax.Array, gloc: jax.Array, dim=3):
    p_loc = p_of_num_multi(loc.shape[1], dim=dim)
    p_glocx = p_of_num_multi(gloc.shape[1], dim=dim)

    imap = get_index_map(p_loc, dim=dim)
    
    gx = []
    for a in range(dim):
        avec = [0] * dim
        avec[a] = 1
        onew = jnp.zeros_like(gloc[:,0])
        for m in iter_multi(p_glocx, dim=dim):
            if (sum(m) > p_glocx) or (sum(m) + sum(avec) > p_loc):
                continue
            b = tuple(m[d] + avec[d] for d in range(dim))
            onew +=  (m[a] + 1) * gloc[:,imap[m]] * loc[:,imap[b]]
        gx.append(onew)
    
    return jnp.stack(gx, axis=-1)
