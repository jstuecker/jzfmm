import jax
import jax.numpy as jnp
import numpy as np
import fmdj_cuda.ffi_multipoles as ffi_multipoles
from .data import TreePlane, Multipoles, PosMass
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

def summarize_multipoles(
        ispl: jnp.ndarray,
        xnode: jnp.ndarray,
        xchild: jnp.ndarray,
        mp: jnp.ndarray, 
        *, cfg: Config, 
        block_size=32) -> Multipoles:
    """Summarizes multipoles from child nodes to parent nodes"""
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

    return Multipoles(xcent=xnode, values=mpnew)
summarize_multipoles.jit = jax.jit(summarize_multipoles, static_argnames=['cfg', 'block_size'])

def multipoles_from_particles(tp: TreePlane, part: PosMass, *, cfg: Config,
                              xcent: jnp.ndarray | None = None) -> Multipoles:
    if xcent is None:
        if cfg.fmm.multipoles_around_com:
            xcent = center_of_mass(tp.ispl, part, cfg=cfg).pos
        else:
            xcent = tp.geom_cent
    
    # particles are monopoles, we can use the same function as for "normal" m2m translation
    mp = summarize_multipoles(
        tp.ispl, xcent, part.pos, part.mass.reshape(-1,1), cfg=cfg
    )

    return mp
multipoles_from_particles.jit = jax.jit(multipoles_from_particles, static_argnames=['cfg'])


def coarsen_multipoles(mp: Multipoles, tp: TreePlane, *, cfg: Config, 
                       xcent: jnp.ndarray | None = None) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
    if xcent is None:
        if cfg.fmm.multipoles_around_com:
            xcent = center_of_mass(tp.ispl, PosMass(pos=mp.xcent, mass=mp.values[:,0]), cfg=cfg).pos
        else:
            xcent = tp.geom_cent

    return summarize_multipoles(
        tp.ispl, xcent, mp.xcent, mp.values, cfg=cfg
    )
coarsen_multipoles.jit = jax.jit(coarsen_multipoles, static_argnames=['cfg'])

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