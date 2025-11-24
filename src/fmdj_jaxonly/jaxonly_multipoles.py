from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import fmdj
from fmdj.config import Config
from fmdj.data import TreePlane, Multipoles, PosMass
from fmdj.multipoles import binomial, define_index_maps, p_to_ncomb, multipole_powers, fact, iterate_multi_indices


def shift_multipoles(m, x0, p=2):
    """m[...,i] corresponds to the expectation value of x**c[0] * y**c[1] * z**c[2]
    we shift it so that the new coefficients hold at the center dx
    mnew[...,i] = (x + x0)**c[0] * (y + y0)**c[1] * (z + z0)**c[2]"""
    combs, index_of_mp = define_index_maps(p)
    x0 = -x0 # Makes writing things easier
    m = m.T
    
    # Stage 1: shift in x
    mx = []
    for a, b, c in combs:
        mnew = 0.
        for i in range(a+1):
            idx_src = index_of_mp[i, b, c]
            coeff = binomial[a, i].astype(m.dtype)
            mnew = mnew + coeff * x0[..., 0]**(a - i) * m[idx_src]
        mx.append(mnew)

    # Stage 2: shift in y
    mxy = []
    for a, b, c in combs:
        mnew = 0.
        for j in range(b+1):
            idx_src = index_of_mp[a, j, c]
            coeff = binomial[b, j].astype(m.dtype)
            mnew = mnew + coeff * x0[..., 1]**(b - j) * mx[idx_src]
        mxy.append(mnew)

    # Stage 3: shift in z
    mxyz = []
    for a, b, c in combs:
        mnew = 0.
        for k in range(c+1):
            idx_src = index_of_mp[a, b, k]
            coeff = binomial[c, k].astype(m.dtype)
            mnew = mnew + coeff * x0[..., 2]**(c - k) * mxy[idx_src]
        mxyz.append(mnew)

    return jnp.stack(mxyz, axis=-1)

def x_moment(x, c):
    return x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

def coarsen_multipoles_jax(mp: Multipoles, tp: TreePlane, *, cfg: Config) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
    parent = tp.icoarse_of_fine()
    kwargs = dict(
        segment_ids=parent,
        num_segments=tp.size(),
        indices_are_sorted=True
    )

    # Compute the center of mass
    mnode = jax.ops.segment_sum(mp.get(0), **kwargs)

    dx = mp.center() - tp.geom_center()[parent]
    mxnode = [jax.ops.segment_sum(dx[...,d]*mp.get(0), **kwargs) for d in range(3)]
    
    if mp.around_com:
        xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1) + tp.geom_center()
    else:
        xcent = tp.geom_center()
    
    dx = xcent[parent] - mp.center()
    
    mpshift = shift_multipoles(mp.values, dx, p=mp.p)

    mp_coarse = [jax.ops.segment_sum(mpshift[...,k], **kwargs) for k in range(mpshift.shape[-1])]

    return Multipoles(
        xcent=xcent,
        values=jnp.stack(mp_coarse, axis=-1),
        p=mp.p,
        around_com=mp.around_com
    )

def multipoles_from_particles_jax(tp: TreePlane, part: PosMass, *, cfg: Config) -> Multipoles:
    dtype = part.mass.dtype

    parent = tp.icoarse_of_fine()
    kwargs = dict(
        segment_ids=parent,
        num_segments=tp.size(),
        indices_are_sorted=True
    )

    # Compute the center of mass
    mnode = jax.ops.segment_sum(part.mass, **kwargs)

    dx = part.pos - tp.geom_center()[parent]
    mxnode = [jax.ops.segment_sum(dx[...,d]*part.mass, **kwargs) for d in range(3)]

    mp = [mnode]
    
    if cfg.fmm.multipoles_around_com:
        xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1) + tp.geom_center()
        mp.extend([jnp.zeros_like(mnode, dtype=dtype)]*3)
    else:
        xcent = tp.geom_center()
        mp.extend(mxnode)
    dx = part.pos - xcent[parent]

    # Compute multipole moments
    for i, nvec in iterate_multi_indices(cfg.fmm.p, istart=4):
        mp.append(jax.ops.segment_sum(x_moment(dx, nvec) * part.mass, **kwargs))

    return Multipoles(
        xcent=xcent,
        values=jnp.stack(mp, axis=-1),
        p=cfg.fmm.p,
        around_com=cfg.fmm.multipoles_around_com
    )

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
    comb, index_map = define_index_maps(p)

    # Precompute Parent relations for the recursive update
    p1s, p2s = np.zeros((len(comb),), dtype=np.int32), np.zeros((len(comb),), dtype=np.int32)
    imaxs = np.zeros((len(comb),), dtype=np.int32)

    w2s = np.zeros((len(comb),), dtype=x.dtype)
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

            imax = p_to_ncomb[p-q]
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
    combs, index_of_mp = define_index_maps(p)
    D = get_Dn(-dx, p=p, eps=eps)

    nks = len(combs)

    # We can precompute the weights and indices we need
    # this way we can map our operation onto a simple matrix multiplication
    # Note that this gives ~ a factor two overhead, because the actual matrix is triangular,
    # since not all multipoles contribute to all orders
    indices = np.zeros((nks,nks), dtype=np.int32)
    weights = np.zeros((nks,nks), dtype=dx.dtype)

    for i,ks in enumerate(combs):
        nvecs = multipole_powers(p-np.sum(ks))

        for j, ns in enumerate(nvecs):
            indices[i,j] = index_of_mp[ks[0]+ns[0], ks[1]+ns[1], ks[2]+ns[2]]
            weights[i,j] = - (-1.)**np.sum(ks) / (fact[ns[0]] * fact[ns[1]] * fact[ns[2]] * fact[ks[0]] * fact[ks[1]] * fact[ks[2]])

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