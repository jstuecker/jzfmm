from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import fmdj
from fmdj.config import Config
from fmdj.data import TreePlane, PosMass
from fmdj.multipoles import num_multi, p_of_num_multi, iter_multi, get_index_map
from fmdj.tools import inverse_of_splits

# ------------------------------------------------------------------------------------------------ #
#                                        Some Combinatorics                                        #
# ------------------------------------------------------------------------------------------------ #

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
#                                           M2M Operators                                          #
# ------------------------------------------------------------------------------------------------ #

def shift_multipoles(m, x0):
    """m[...,i] corresponds to the expectation value of x**c[0] * y**c[1] * z**c[2]
    we shift it so that the new coefficients hold at the center dx
    mnew[...,i] = (x + x0)**c[0] * (y + y0)**c[1] * (z + z0)**c[2]"""
    p = p_of_num_multi(m.shape[-1])

    index_of_mp = get_index_map(p)
    x0 = -x0 # Makes writing things easier
    m = m.T
    
    # Stage 1: shift in x
    mx = []
    for a, b, c in iter_multi(p):
        mnew = 0.
        for i in range(a+1):
            mnew = mnew + binom(a, i) * x0[..., 0]**(a - i) * m[index_of_mp[i, b, c]]
        mx.append(mnew)

    # Stage 2: shift in y
    mxy = []
    for a, b, c in iter_multi(p):
        mnew = 0.
        for j in range(b+1):
            mnew = mnew + binom(b, j) * x0[..., 1]**(b - j) * mx[index_of_mp[a, j, c]]
        mxy.append(mnew)

    # Stage 3: shift in z
    mxyz = []
    for a, b, c in iter_multi(p):
        mnew = 0.
        for k in range(c+1):
            mnew = mnew + binom(c, k) * x0[..., 2]**(c - k) * mxy[index_of_mp[a, b, k]]
        mxyz.append(mnew)

    return jnp.stack(mxyz, axis=-1)

def x_moment(x, c):
    return x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

def coarsen_multipoles_jax(mp: jax.Array, tp: TreePlane, *, cfg: Config) -> jax.Array:
    """Determines the multipoles at the next coarser tree plane"""
    parent = inverse_of_splits(tp.ispl, mp.size)
    kwargs = dict(
        segment_ids=parent,
        num_segments=tp.size(),
        indices_are_sorted=True
    )

    # Compute the center of mass
    mnode = jax.ops.segment_sum(mp[0], **kwargs)

    dx = mp.center() - tp.geom_cent[parent]
    mxnode = [jax.ops.segment_sum(dx[...,d]*mp[0], **kwargs) for d in range(3)]
    
    if cfg.fmm.multipoles_around_com:
        xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1) + tp.geom_cent
    else:
        xcent = tp.geom_cent
    
    dx = xcent[parent] - mp.center()
    
    mpshift = shift_multipoles(mp, dx)

    mp_coarse = [jax.ops.segment_sum(mpshift[...,k], **kwargs) for k in range(mpshift.shape[-1])]

    return jnp.stack(mp_coarse, axis=-1)
coarsen_multipoles_jax.jit = jax.jit(coarsen_multipoles_jax, static_argnames=['cfg'])

def multipoles_from_particles_jax(tp: TreePlane, part: PosMass, *, cfg: Config) -> jax.Array:
    dtype = part.mass.dtype

    parent = inverse_of_splits(tp.ispl, mp.size)
    kwargs = dict(
        segment_ids=parent,
        num_segments=tp.size(),
        indices_are_sorted=True
    )

    # Compute the center of mass
    mnode = jax.ops.segment_sum(part.mass, **kwargs)

    dx = part.pos - tp.geom_cent[parent]
    mxnode = [jax.ops.segment_sum(dx[...,d]*part.mass, **kwargs) for d in range(3)]

    mp = [mnode]
    
    if cfg.fmm.multipoles_around_com:
        xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1) + tp.geom_cent
        mp.extend([jnp.zeros_like(mnode, dtype=dtype)]*3)
    else:
        xcent = tp.geom_cent
        mp.extend(mxnode)
    dx = part.pos - xcent[parent]

    # Compute multipole moments
    for nvec in iter_multi(cfg.fmm.p, istart=4):
        mp.append(jax.ops.segment_sum(x_moment(dx, nvec) * part.mass, **kwargs))

    return np.stack(mp, axis=-1)
multipoles_from_particles_jax.jit = jax.jit(multipoles_from_particles_jax, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                                   Green's Function Derivatives                                   #
# ------------------------------------------------------------------------------------------------ #

def get_gs(x, nmax=1, eps=0.):
    """These are the derivatives (1/r d/dr)^n (1/r)"""
    rinv = 1. / jnp.sqrt(jnp.sum(x**2, axis=-1) + eps*eps)
    r2inv = rinv * rinv
    gs = [rinv]
    for n in range(nmax):
        gs.append(-(1+2*n) * gs[-1] * r2inv)

    return gs

def get_Dn(x, p=0, eps=0.):
    """Get the all the derivatives of a Green's function g
    Using the recurrence relation from Tausch (2003)
    http://dx.doi.org/10.1090/conm/329/05866 (See Section 3)
    Holds for arbitrary radial Green's functions (also non-harmonic cases)

    We want to compute D(n)G(0). The recurrence relates D(n)G(q) to D(n-1)G(q+1) and D(n-2)G(q+1)
    """
    # comb, index_map = define_index_maps(p)
    index_map = get_index_map(p)
    comb = tuple(iter_multi(p))
    ncomb = len(comb)

    # Precompute Parent relations for the recursive update
    p1s, p2s = np.zeros((ncomb,), dtype=np.int32), np.zeros((ncomb,), dtype=np.int32)
    imaxs = np.zeros((ncomb,), dtype=np.int32)

    w2s = np.zeros((ncomb,), dtype=x.dtype)
    for i, nvec in enumerate(comb):
        imax = np.argmax(nvec, axis=-1)
        nmax = nvec[imax]
        p1, p2 = np.copy(nvec), np.copy(nvec)
        p1[imax] = np.clip(nmax - 1, 0, None)
        p2[imax] = np.clip(nmax - 2, 0, None)

        p1s[i] = index_map[*p1]
        p2s[i] = index_map[*p2]

        w2s[i] = (nmax - 1)
        imaxs[i] = imax

    def get_Dn(x):
        gs = get_gs(x, p+1, eps=eps)
        w1s = x[imaxs]

        Dn = gs[p][None]
        for q in range(p-1, -1, -1):
            # Apply the recurrence relation iteratively. Dn grows in every step.

            imax = num_multi(p-q)
            D1 = w1s[1:imax] * Dn[p1s[1:imax]]
            D2 = w2s[1:imax] * Dn[p2s[1:imax]]
            Dn = jnp.concatenate((gs[q][None], D1 + D2))

        return jnp.stack(Dn)
    return jax.vmap(get_Dn, in_axes=(0,), out_axes=0)(x)
get_Dn.jit = jax.jit(get_Dn, static_argnames=("p", "eps"))


# ------------------------------------------------------------------------------------------------ #
#                                                M2L                                               #
# ------------------------------------------------------------------------------------------------ #

def single_multipole_to_local(mp, dx, p=2, eps=0.):
    """Returns the expansion coefficients for the interaction between two nodes"""
    combs = tuple(iter_multi(p))
    index_of_mp = get_index_map(p)

    D = get_Dn(-dx, p=p, eps=eps)

    nks = len(combs)

    # We can precompute the weights and indices we need
    # this way we can map our operation onto a simple matrix multiplication
    # Note that this gives ~ a factor two overhead, because the actual matrix is triangular,
    # since not all multipoles contribute to all orders
    indices = np.zeros((nks,nks), dtype=np.int32)
    weights = np.zeros((nks,nks), dtype=dx.dtype)

    f = [fact(i) for i in range(p+1)]

    for i,ks in enumerate(combs):
        for j, ns in enumerate(iter_multi(p - np.sum(ks))):
            indices[i,j] = index_of_mp[ks[0]+ns[0], ks[1]+ns[1], ks[2]+ns[2]]
            weights[i,j] = - (-1.)**np.sum(ks) / (f[ns[0]] * f[ns[1]] * f[ns[2]] * f[ks[0]] * f[ks[1]] * f[ks[2]])

    Lk = jnp.einsum("...ij,...j,ij->...i", D[...,indices], mp, weights)

    return Lk

def _reduce_fsum_chunked(f, y0, x, istart, iend, chunk_size):
    """
    Applies `f(x,mask)` to chunks of chunk_size over `x[istart:iend]` and sums the results.

    Parameters:
    - f: function taking x_chunk, mask -> output of fixed shape
    - x: [n, ...] input array (larger than or equal to nmax)
    - istart: start in the array
    - iend: end in the array (exclusive)
    - chunk_size: number of elements per chunk

    Returns:
    - total: accumulated result from all chunks
    """
    iarange = jnp.arange(chunk_size, dtype=jnp.int32)
    num_chunks = (iend-istart + chunk_size - 1) // chunk_size # corresponds to ceil((iend-istart) / chunk_size)

    def body(i, y):
        i0 = istart + i * chunk_size
        return f(y, x[i0 + iarange], i0 + iarange < iend)

    return jax.lax.fori_loop(0, num_chunks, body, y0)

def ilist_node_to_node(xnodes, multipoles, interactions, irange, cfg : Config):
    p = cfg.fmm.p

    chunk_size = int((cfg.old.ilist_max_mb * 1024**2) // (2 * xnodes.dtype.itemsize * ((p+3) * (p+2) * (p+1) / 6)**2))
    chunk_size = min(max((chunk_size//64)*64,  64), len(xnodes)*4)

    loc = jnp.zeros(multipoles.shape, dtype=xnodes.dtype)

    def eval_node_node(loc, iab, mask):
        weights = single_multipole_to_local(multipoles[iab[:,1]], xnodes[iab[:,0]] - xnodes[iab[:,1]], p=p, eps=cfg.softening)
        weights = jnp.where(mask[:,None], weights, 0.)
        return loc.at[iab[:,0]].add(weights)
    loc = _reduce_fsum_chunked(eval_node_node, loc, interactions, irange[0], irange[1], chunk_size=chunk_size)

    return loc

# ------------------------------------------------------------------------------------------------ #
#                                      Local 2 Local operators                                     #
# ------------------------------------------------------------------------------------------------ #

def shift_local_to_local_jax(L, dx):
    L = L.T
    p = p_of_num_multi(L.shape[0])
    index_of_mp = get_index_map(p)
    
    # Stage 1: shift in x
    Lx = []
    for a, b, c in iter_multi(p):
        Lnew = 0.
        for i in range(a, p+1-b-c):
            Lnew = Lnew + binom(i, a) * dx[..., 0]**(i - a) * L[index_of_mp[(i, b, c)]]
        Lx.append(Lnew)

    # Stage 2: shift in y
    Lxy = []
    for a, b, c in iter_multi(p):
        Lnew = 0.
        for j in range(b, p+1-a-c):
            Lnew = Lnew + binom(j, b) * dx[..., 1]**(j - b) * Lx[index_of_mp[(a, j, c)]]
        Lxy.append(Lnew)

    # Stage 3: shift in z
    Lxyz = []
    for a, b, c in iter_multi(p):
        Lnew = 0.
        for k in range(c, p+1-a-b):
            Lnew = Lnew + binom(k, c) * dx[..., 2]**(k - c) * Lxy[index_of_mp[(a, b, k)]]
        Lxyz.append(Lnew)

    return jnp.stack(Lxyz, axis=-1)

def evaluate_local_fphi_jax(L, x):
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
evaluate_local_fphi_jax.jit = jax.jit(evaluate_local_fphi_jax)