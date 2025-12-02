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

def iter_multi(p, istart=0):
    i = 0
    for n in range(p+1):
        for nz in range(n+1):
            for ny in range(n - nz +1):
                nx = n - ny - nz
                if i >= istart:
                    yield nx, ny, nz
                i += 1

def get_index_map(p):
    return {c: i for i,c in enumerate(iter_multi(p))}

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

def _summarize_multipoles_impl(ispl, mp, xnode, xchild, *, cfg, block_size=32):
    """Summarizes multipoles from child nodes to parent nodes"""
    if len(mp.shape) == 1: # probably plain masses corresponding to monopoles
        mp = mp.reshape(-1,1)

    assert mp.dtype == jnp.float32
    assert ispl.dtype == jnp.int32
    dtype = mp.dtype
    p_in = p_of_num_multi(mp.shape[1])
    out_mp = jax.ShapeDtypeStruct((ispl.size-1, num_multi(cfg.fmm.p)), dtype)
    mpnew = jax.ffi.ffi_call("SummarizeMultipoles", (out_mp,))(
        ispl, mp, xnode, xchild,
        p_in=np.int32(p_in),
        p=np.int32(cfg.fmm.p),
        block_size=np.uint64(block_size),
        kahan = cfg.fmm.kahan_summation
    )[0]

    return mpnew

def summarize_multipoles(
        ispl: jnp.ndarray,
        mp: jnp.ndarray,
        xnode: jnp.ndarray,
        xchild: jnp.ndarray,
        *, cfg: Config
    ) -> jnp.ndarray:

    if mp.ndim == 1:
        mp = mp.reshape(-1,1)
    pin = p_of_num_multi(mp.shape[-1])

    @jax.custom_gradient
    def inner(xchild, mp):

        mp_n = _summarize_multipoles_impl(ispl, mp, xnode, xchild, cfg=cfg)

        def grad(gmp_n):
            gmp = shift_local_to_children(ispl, gmp_n, xnode, xchild, cfg=cfg, pout=max(pin, 1))
            gx = gmp[..., 1:4] * mp[..., 0:1]

            return gx, gmp[:,:num_multi(pin)]
        
        return mp_n, grad
    
    return inner(xchild, mp)
summarize_multipoles.jit = jax.jit(summarize_multipoles, static_argnames=['cfg', ])

def build_multipole_hierarchy(th: list[TreePlane], pos: jnp.ndarray, mp: jnp.ndarray, *, cfg: Config
                              ) -> list[jnp.ndarray]:
    if len(mp.shape) == 1:
        mp = mp.reshape(-1,1)

    mp0 = summarize_multipoles(th[0].ispl, mp, th[0].center(), pos, cfg=cfg)
    mph = [mp0]
    for i in range(1, len(th)):
        mp_coarse = summarize_multipoles(
            th[i].ispl, mph[-1], th[i].center(), th[i-1].center(), cfg=cfg
        )
        mph.append(mp_coarse)
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

def shift_local_to_children(
        ispl: jnp.array,
        loc: jnp.array,
        xnode: jnp.array,
        xchild: jnp.array,
        cfg: Config,
        pout: int = None
    ) -> jnp.ndarray:

    if loc.ndim == 1:
        loc = loc.reshape(-1,1)
    p = p_of_num_multi(loc.shape[-1])
    if pout is None:
        pout = p_of_num_multi(loc.shape[-1])

    @jax.custom_vjp
    def eval(xchild, loc):
        return _shift_local_to_children_impl(ispl, loc, xnode, xchild, pout=min(pout,p))
    
    def eval_fwd(xchild, loc): # save higher order local expansion for the backward pass
        assert pout + 1 <= p, "have not implemented this scenario yet"
        loc_c = _shift_local_to_children_impl(ispl, loc, xnode, xchild, pout=min(pout+1,p))
        return loc_c[...,:num_multi(pout)], (loc_c, xchild)
    
    def eval_bwd(res, gloc_c): # the adjoint of the l2l operator is an m2m operator
        loc_c, xchild = res
        gloc = summarize_multipoles(ispl, gloc_c, xnode, xchild, cfg=cfg)
        gx = local_eval_vjp(loc_c, gloc_c)
        return gx, gloc
    
    eval.defvjp(eval_fwd, eval_bwd)
    
    return eval(xchild, loc)
shift_local_to_children.jit = jax.jit(shift_local_to_children, static_argnames=["cfg", "pout"])

def local_eval_vjp(loc: jnp.ndarray, gloc: jnp.ndarray):
    p_loc, p_gloc = p_of_num_multi(loc.shape[1]), p_of_num_multi(gloc.shape[1])

    assert p_loc == p_gloc + 1

    imap = get_index_map(p_loc)
    
    out = []
    for a in range(3):
        avec = [0,0,0]
        avec[a] = 1
        onew = jnp.zeros_like(gloc[:,0])
        for m in iter_multi(p_gloc):
            if sum(m) + sum(avec) > p_loc:
                continue
            b = (m[0]+avec[0], m[1]+avec[1], m[2]+avec[2])
            onew +=  (m[a] + 1) * gloc[:,imap[m]] * loc[:,imap[b]]
        out.append(onew)
    
    return jnp.stack(out, axis=-1)