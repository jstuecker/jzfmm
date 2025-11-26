import jax
import jax.numpy as jnp
import numpy as np
import fmdj_cuda.ffi_multipoles as ffi_multipoles
from .data import TreePlane, Multipoles, PosMass
from .config import Config

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

jax.ffi.register_ffi_target("MultipolesFromParticles", ffi_multipoles.MultipolesFromParticles(), platform="CUDA")
jax.ffi.register_ffi_target("CoarsenMultipoles", ffi_multipoles.CoarsenMultipoles(), platform="CUDA")

def multipoles_from_particles(tp: TreePlane, part: PosMass, *, cfg: Config) -> Multipoles:
    assert cfg.fmm.multipoles_around_com

    posm = part.posm()

    assert posm.dtype == jnp.float32
    assert tp.ispl.dtype == jnp.int32

    ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])
    
    out_mp = jax.ShapeDtypeStruct((tp.size(), ncomb[cfg.fmm.p]), posm.dtype)
    out_xcent = jax.ShapeDtypeStruct((tp.size(), 3), posm.dtype)

    mp, xcent = jax.ffi.ffi_call("MultipolesFromParticles", (out_mp, out_xcent))(
        tp.ispl, posm, p=np.int32(cfg.fmm.p), block_size=np.uint64(32)
    )
    
    return Multipoles(xcent=xcent, values=mp, p=cfg.fmm.p, around_com=True)
multipoles_from_particles.jit = jax.jit(multipoles_from_particles, static_argnames=['cfg'])

def coarsen_multipoles(mp: Multipoles, tp: TreePlane, *, cfg: Config) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
    assert mp.around_com

    dtype = mp.values.dtype

    assert mp.values.dtype == jnp.float32
    assert tp.ispl.dtype == jnp.int32

    ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])

    out_mp = jax.ShapeDtypeStruct((tp.size(), ncomb[mp.p]), dtype)
    out_xcent = jax.ShapeDtypeStruct((tp.size(), 3), dtype)

    mpnew, xcent = jax.ffi.ffi_call("CoarsenMultipoles", (out_mp, out_xcent))(
        tp.ispl, mp.values, mp.center(), p=np.int32(mp.p), block_size=np.uint64(32)
    )
    return Multipoles(xcent=xcent, values=mpnew, p=mp.p, around_com=mp.around_com)
coarsen_multipoles.jit = jax.jit(coarsen_multipoles, static_argnames=['cfg'])

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

def multi_to_flat(kx, ky, kz):
    p = kx + ky + kz
    npoff = ((p+2)*(p+1)*p // 6)
    npoff += kz*(2*p + 3 - kz)//2 + ky

    return npoff

def fact(n):
    return int(np.prod(np.arange(1, n+1), initial=1))

def binom(n, k):
    return fact(n) // (fact(k) * fact(n - k))


# ------------------------------------------------------------------------------------------------ #
#                                      Local 2 Local operators                                     #
# ------------------------------------------------------------------------------------------------ #

def shift_local_to_local(L, x0):
    L = L.T
    p = p_of_num_multi(L.shape[0])
    index_of_mp = get_index_map(p)
    
    # Stage 1: shift in x
    Lx = []
    for a, b, c in iter_multi(p):
        Lnew = 0.
        for i in range(a, p+1-b-c):
            Lnew = Lnew + binom(i, a) * x0[..., 0]**(i - a) * L[index_of_mp[(i, b, c)]]
        Lx.append(Lnew)

    # Stage 2: shift in y
    Lxy = []
    for a, b, c in iter_multi(p):
        Lnew = 0.
        for j in range(b, p+1-a-c):
            Lnew = Lnew + binom(j, b) * x0[..., 1]**(j - b) * Lx[index_of_mp[(a, j, c)]]
        Lxy.append(Lnew)

    # Stage 3: shift in z
    Lxyz = []
    for a, b, c in iter_multi(p):
        Lnew = 0.
        for k in range(c, p+1-a-b):
            Lnew = Lnew + binom(k, c) * x0[..., 2]**(k - c) * Lxy[index_of_mp[(a, b, k)]]
        Lxyz.append(Lnew)

    return jnp.stack(Lxyz, axis=-1)

def evaluate_local_fphi(L, x):
    """Evaluates the function value of the expansion at x"""
    p = p_of_num_multi(L.shape[-1])

    phi = 0.
    for index, c in enumerate(iter_multi(p)):
        phi +=  L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

    fx, fy, fz = 0., 0., 0.
    for index, c in enumerate(iter_multi(p)):
        if c[0] > 0:
            fx += - L[...,index] * x[...,0]**(c[0]-1) * x[...,1]**c[1] * x[...,2]**c[2] * c[0]
        if c[1] > 0:
            fy += - L[...,index] * x[...,0]**c[0] * x[...,1]**(c[1]-1) * x[...,2]**c[2] * c[1]
        if c[2] > 0:
            fz += - L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**(c[2]-1) * c[2]
    
    fphi = jnp.stack((fx, fy, fz, phi), axis=-1)

    return fphi
evaluate_local_fphi.jit = jax.jit(evaluate_local_fphi)