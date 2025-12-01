import jax
import jax.numpy as jnp
import numpy as np
import fmdj_cuda.ffi_multipoles as ffi_multipoles
from .data import TreePlane, PosMass
from .config import Config

# ------------------------------------------------------------------------------------------------ #
#                                        Some Combinatorics                                        #
# ------------------------------------------------------------------------------------------------ #

def num_multi(p):
    return (p + 1) * (p + 2) * (p + 3) // 6

def p_of_num_multi(ncomp):
    return [num_multi(p) for p in range(20)].index(ncomp)

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

jax.ffi.register_ffi_target("TranslateLocalToLocal", ffi_multipoles.TranslateLocalToLocal(), platform="CUDA")
jax.ffi.register_ffi_target("CenterOfMass", ffi_multipoles.CenterOfMass(), platform="CUDA")
jax.ffi.register_ffi_target("SummarizeMultipoles", ffi_multipoles.SummarizeMultipoles(), platform="CUDA")

def center_of_mass(ispl: jnp.ndarray, part: PosMass, *, cfg: Config, block_size=32) -> PosMass:
    """Computes the center of mass of the nodes in the tree plane"""
    assert part.pos.dtype == jnp.float32
    assert ispl.dtype == jnp.int32

    out_xcent = jax.ShapeDtypeStruct((ispl.size-1, 4), part.pos.dtype)

    xm = jax.ffi.ffi_call("CenterOfMass", (out_xcent,))(
        ispl, part.pos, part.mass,
        kahan = cfg.fmm.kahan_summation,
        block_size=np.uint64(block_size)
    )[0]
    return PosMass(pos=xm[...,0:3], mass=xm[...,4])
center_of_mass.jit = jax.jit(center_of_mass, static_argnames=['cfg', 'block_size'])

def _summarize_multipoles_impl(ispl, xnode, xchild, mp, *, cfg, block_size=32) -> jnp.ndarray:
    """Summarizes multipoles from child nodes to parent nodes"""
    if len(mp.shape) == 1: # probably plain masses corresponding to monopoles
        mp = mp.reshape(-1,1)

    assert mp.dtype == jnp.float32
    assert ispl.dtype == jnp.int32
    dtype = mp.dtype
    p_in = p_of_num_multi(mp.shape[1])
    out_mp = jax.ShapeDtypeStruct((ispl.size-1, num_multi(cfg.fmm.p)), dtype)
    mpnew = jax.ffi.ffi_call("SummarizeMultipoles", (out_mp,))(
        ispl, xnode, xchild, mp,
        p_in=np.int32(p_in),
        p=np.int32(cfg.fmm.p),
        block_size=np.uint64(block_size),
        kahan = cfg.fmm.kahan_summation
    )[0]

    return mpnew

def summarize_multipoles(
        ispl: jnp.ndarray,
        xnode: jnp.ndarray,
        xchild: jnp.ndarray,
        mp: jnp.ndarray, 
        *, cfg: Config, 
        block_size=32
    ) -> jnp.ndarray:

    pin = p_of_num_multi(mp.shape[-1])

    if mp.ndim == 1:
        mp = mp.reshape(-1,1)

    @jax.custom_gradient
    def inner(xchild, mp):

        val = _summarize_multipoles_impl(ispl, xnode, xchild, mp, cfg=cfg, block_size=block_size)

        def grad(gmp):
            gmp = shift_local_to_children(ispl, gmp, xnode, xchild, pout=max(pin, 1), block_size=block_size)
            gx = gmp[..., 1:4] * mp[..., 0:1]

            return gx, gmp[:,:num_multi(pin)]
        
        return val, grad
    
    return inner(xchild, mp)
summarize_multipoles.jit = jax.jit(summarize_multipoles, static_argnames=['cfg', 'block_size'])

def build_multipole_hierarchy(th: list[TreePlane], pos: jnp.ndarray, mp: jnp.ndarray, *, cfg: Config
                              ) -> list[jnp.ndarray]:
    if len(mp.shape) == 1:
        mp = mp.reshape(-1,1)

    mp0 = summarize_multipoles(th[0].ispl, th[0].center(), pos, mp, cfg=cfg)
    mph = [mp0]
    for i in range(1, len(th)):
        mp_coarse = summarize_multipoles(
            th[i].ispl, th[i].center(), th[i-1].center(), mph[-1], cfg=cfg
        )
        mph.append(mp_coarse)
    return mph
build_multipole_hierarchy.jit = jax.jit(build_multipole_hierarchy, static_argnames=['cfg'])

def shift_local_to_children(
        ispl: jnp.array,
        loc: jnp.array,
        xnode: jnp.array,
        xchild: jnp.array,
        pout=None,
        block_size=32
    ) -> jnp.array:
    """Shifts local expansions to child nodes"""
    dtype = loc.dtype

    assert loc.dtype == jnp.float32

    p = p_of_num_multi(loc.shape[1])

    if pout is None:
        pout = p

    assert (pout >= 0) and (pout <= p)

    out_loc = jax.ShapeDtypeStruct((xchild.shape[0], num_multi(pout)), dtype)

    locnew = jax.ffi.ffi_call("TranslateLocalToLocal", (out_loc,))(
        ispl, loc, xnode, xchild,
        p=np.int32(p), pout=np.int32(pout), block_size=np.uint64(block_size)
    )[0]
    return locnew
shift_local_to_children.jit = jax.jit(shift_local_to_children, static_argnames=['pout', 'block_size'])

from fmdj_jaxonly.jaxonly_multipoles import get_index_map, iter_multi

def local_eval_vjp(loc: jnp.ndarray, gloc: jnp.ndarray):
    p_loc, p_gloc = p_of_num_multi(loc.shape[1]), p_of_num_multi(gloc.shape[1])

    assert p_loc == p_gloc + 1

    imap_a, imap_b = get_index_map(p_loc), get_index_map(p_gloc)
    
    out = []
    for a1,a2,a3 in ((1,0,0),(0,1,0),(0,0,1)):
        onew = jnp.zeros_like(gloc[:,0])
        for m1, m2, m3 in iter_multi(p_gloc):
            if m1 + a1 + m2 + a2 + m3 + a3 > p_loc:
                continue
            onew += gloc[:,imap_b[m1,m2,m3]] * loc[:,imap_a[m1+a1,m2+a2,m3+a3]]
        out.append(onew)
    
    return jnp.stack(out, axis=-1)