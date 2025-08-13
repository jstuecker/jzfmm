import jax
import jax.numpy as jnp
from .octree import Octree
import numpy as np
from math import factorial
from functools import lru_cache

try:
    import custom_jax as cj
except ImportError:
    print("No custom JAX found, using fall-back solutions. This may significantly degrade performance.")
    cj = None

# ============================= Some fixed Combinatorical Computations =========================== #

def generate_combinations(p):
    """Generate unique triples (i, j, k) such that i + j + k = p."""
    combos = []
    for i in range(p + 1):
        for j in range(p + 1 - i):
            k = p - i - j
            combos.append((k, j, i))
    return combos

fact = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880])
binomial = np.zeros((6, 6), dtype=int)
for n in range(6):
    for k in range(n + 1):
        binomial[n, k] = fact[n] // (fact[k] * fact[n - k])

rank_to_ncomp = np.array([1, 3, 6, 10, 15, 21, 28])
p_to_ncomb = np.array([1, 4, 10, 20, 35, 56, 84])
# ncomp_to_rank = {1: 0, 3: 1, 6: 2, 10: 3, 15: 4, 21: 5, 28: 6, 36: 7}
# ncomb_to_p = {1: 0, 4: 1, 10: 2, 20: 3, 35: 4, 56: 5, 84: 6, 120: 7}
ncomb_to_p = np.zeros(p_to_ncomb[-1] + 1, dtype=np.int32)
ncomb_to_p[p_to_ncomb] = np.arange(len(p_to_ncomb))
ncomp_to_rank = np.zeros(rank_to_ncomp[-1] + 1, dtype=np.int32)
ncomp_to_rank[rank_to_ncomp] = np.arange(len(rank_to_ncomp))

combinations = np.concatenate([generate_combinations(i) for i in range(7)])
index_map = np.zeros((7, 7, 7), dtype=np.int32) - 1
for index, c in enumerate(combinations):
    index_map[c[0], c[1], c[2]] = index

def multipole_powers(p : int):
    """Generate unique triples (i, j, k) such that i + j + k <= p."""
    return combinations[:p_to_ncomb[p]]

def define_index_maps(p):
    return combinations[:p_to_ncomb[p]], index_map[:p + 1, :p + 1, :p + 1]

# =============================== Multipole to Multipole  Operators ============================== #

def save_divide(a, b):
    return jnp.where(b != 0, a / b, 0.)

def com_via_height(octree : Octree, pos, mass):
    """Computes the mass and the center of mass of each node in the octree"""
    max_nodes = len(octree.lchild)

    # First add particles into their parent nodes
    m = jnp.zeros(max_nodes, dtype=jnp.float32
                  ).at[octree.node_of_particle].add(mass)
    mx = jnp.zeros((max_nodes, 3), dtype=jnp.float32
                   ).at[octree.node_of_particle].add(pos * mass[:,None])

    # Next propagate information up the tree
    height_parent = octree.height[octree.parent]
    def handle_height_level(hlvl, carry):
        m, mx = carry

        sel = (height_parent == hlvl) & octree.is_valid
        ipar = jnp.where(sel, octree.parent, max_nodes)
        
        m = m.at[ipar].add(m)
        mx = mx.at[ipar].add(mx)

        return m, mx
    
    m, mx = jax.lax.fori_loop(2, octree.maxheight+1, handle_height_level, (m, mx))

    return m, save_divide(mx, m[:,None])
com_via_height.jit = jax.jit(com_via_height)

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
            idx_dst = index_of_mp[a, b, c]
            coeff = binomial[a, i]
            mnew = mnew + coeff * x0[..., 0]**(a - i) * m[idx_src]
        mx.append(mnew)

    # Stage 2: shift in y
    mxy = []
    for a, b, c in combs:
        mnew = 0.
        for j in range(b+1):
            idx_src = index_of_mp[a, j, c]
            coeff = binomial[b, j]
            mnew = mnew + coeff * x0[..., 1]**(b - j) * mx[idx_src]
        mxy.append(mnew)

    # Stage 3: shift in z
    mxyz = []
    for a, b, c in combs:
        mnew = 0.
        for k in range(c+1):
            idx_src = index_of_mp[a, b, k]
            coeff = binomial[c, k]
            mnew = mnew + coeff * x0[..., 2]**(c - k) * mxy[idx_src]
        mxyz.append(mnew)

    return jnp.stack(mxyz, axis=-1)

def x_moment(x, c):
    return x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

def multipoles_via_height(octree : Octree, pos, mass, p=2, xcom=None):
    x0 = xcom if xcom is not None else jnp.zeros((octree.max_nodes, 3), dtype=jnp.float32)

    max_nodes = len(octree.lchild)
    comb = multipole_powers(p)
    parent_of_part = octree.node_of_particle
    
    # We make an array for each multipole moment, this way we can avoid copying the others on each individual update
    mp = []

    # First, we add each particle to its parent node
    for i,c in enumerate(comb):
        mppart = x_moment(pos - x0[parent_of_part], c) * mass

        mp.append(jax.ops.segment_sum(mppart, parent_of_part, num_segments=max_nodes, indices_are_sorted=True))

    mp = jnp.stack(mp, axis=-1)

    # Next we need to propagate multipoles up the tree
    height_parent = octree.height[octree.parent]
    dxparent = x0[octree.parent] - x0
    def handle_height_level(hlvl, mp):
        sel = (octree.height[octree.parent] == hlvl) & octree.is_valid
        ipar = jnp.where(sel, octree.parent, max_nodes)

        if xcom is not None:
            mpnew = shift_multipoles(mp, dxparent, p=p)
        else:
            mpnew = mp
        
        return mp.at[ipar].add(mpnew)
    
    mp = jax.lax.fori_loop(2, octree.maxheight+1, handle_height_level, mp)

    return mp
multipoles_via_height.jit = jax.jit(multipoles_via_height, static_argnames=("p",))

# =================================== Local to Local  Operators ================================== #
def shift_local_to_local(L, dx):
    """To shift from expansion-coefficients around x0 to expansion around x1 put dx = x1-x0"""
    p = ncomb_to_p[L.shape[-1]]
    combs, index_of_mp = define_index_maps(p)
    Lout = []
    for index, c in enumerate(combs):
        Lnew = 0.
        for i in range(c[0], p+1):
            for j in range(c[1], p+1-i):
                for k in range(c[2], p+1-i-j):
                    bfac = binomial[i, c[0]] * binomial[j, c[1]] * binomial[k, c[2]]
                    Lnew = Lnew + bfac * L[...,index_of_mp[i,j,k]] * dx[...,0]**(i-c[0]) * dx[...,1]**(j-c[1]) * dx[...,2]**(k-c[2])
        Lout.append(Lnew)

    return jnp.stack(Lout, axis=-1)

def local_to_local_via_height(octree : Octree, Lk):
    """Shifts local expansion coefficents down the tree"""
    height_parent = octree.height[octree.parent]
    dxparent = octree.xnode - octree.xnode[octree.parent]

    # Next we need to propagate multipoles up the tree
    def handle_height_level(i, Lk):
        hlvl = -i
        sel = (height_parent == hlvl) & octree.is_valid

        Lknew = shift_local_to_local(Lk[octree.parent], dxparent)
        
        return jnp.where(sel[:,None], Lk + Lknew, Lk)

    return jax.lax.fori_loop(-octree.maxheight, 0, handle_height_level, Lk)
local_to_local_via_height.jit = jax.jit(local_to_local_via_height)

def evaluate_local(L, x):
    """Evaluates the function value of the expansion at x"""
    p = ncomb_to_p[L.shape[-1]]
    combs, index_of_mp = define_index_maps(p)

    phi = 0.
    for index, c in enumerate(combs):
        phi +=  L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

    return phi
evaluate_local.jit = jax.jit(evaluate_local)

# ============================= Tree build convenience functions ================================= #

def calculate_multipoles_for_tree(octree : Octree, pos, mass, p=2) -> Octree:
    """Calculates octree.mp and octree.xnode"""
    
    octree.p = p
    m, octree.xnode = com_via_height(octree, pos, mass)
    octree.mp = multipoles_via_height(octree, pos, mass, p=p, xcom=octree.xnode)
    
    return octree

# ================================ Potential Helper Functions ==================================== #

def _get_gs(x, nmax=1, eps=0.):
    """These are the derivatives (1/r d/dr)^n (1/r)"""
    r = jnp.sqrt(jnp.sum(x**2, axis=-1) + eps**2)
    
    gs = []

    if nmax >= 0: gs.append(1./r)
    if nmax >= 1: gs.append(-1./r**3)
    if nmax >= 2: gs.append(3./r**5)
    if nmax >= 3: gs.append(-15./r**7)
    if nmax >= 4: gs.append(105./r**9)
    if nmax >= 5: gs.append(-945./r**11)
    if nmax >= 6: gs.append(10395./r**13)
    if nmax >= 7: gs.append(135135./r**15)

    return gs

def _get_gs_v2(x, nmax=1, eps=0.):
    """These are the derivatives (1/r d/dr)^n (1/r)"""
    rinv = 1. / jnp.sqrt(jnp.sum(x**2, axis=-1) + eps*eps)
    r2inv = rinv * rinv
    gs = [rinv]
    for n in range(nmax):
        gs.append(-(1+2*n) * gs[-1] * r2inv)

    return gs

def _get_Dn(x, g, nx=0, ny=0, nz=0):
    """Dn = nabla^n (1/r)
          = (1/r d/dr)^n g0(r) * x^nx * y^ny * z^nz"""
    n = nx + ny + nz

    ni = np.array([nx, ny, nz], dtype=np.int64)
    isort = np.argsort(ni)[::-1]
    nsort = np.array([nx, ny, nz])[isort]
    xpow = x[...,0]**nx * x[...,1]**ny * x[...,2]**nz

    # This function checks the signature of nx,ny,nz
    def sig(nx, ny=-1, nz=-1):
        if ny == -1: # only compare nx
            return nsort[0] == nx
        elif nz == -1:
            return (nsort[0] == nx) & (nsort[1] == ny)
        else:
            return (nsort[0] == nx) & (nsort[1] == ny) * (nsort[2] == nz)

    if n == 0:
        return g[0]
    elif n == 1:
        return g[1]*x[...,isort[0]]
    elif n == 2:
        return g[1]*sig(2) + g[2]*xpow
    elif n == 3:
        return g[2]*(3*sig(3)*x[...,isort[0]] + sig(2,1)*x[...,isort[1]]) + g[3]*xpow
    elif n == 4:
        return (g[2]*(3*sig(4) + sig(2,2)) 
                + g[3]*(6*sig(4)*x[...,isort[0]]**2 + 3*sig(3,1)*x[...,isort[0]]*x[...,isort[1]] + sig(2,2)*(x[...,isort[0]]**2 + x[...,isort[1]]**2) + sig(2,1,1)*x[...,isort[1]]*x[...,isort[2]])
                + g[4]*xpow)
    elif n == 5:
        return (g[3]*(15*sig(5)*x[...,isort[0]] + 3*sig(4,1)*x[...,isort[1]] + 3*sig(3,2)*x[...,isort[0]] + sig(2,2,1)*x[...,isort[2]]) 
                + g[4]*(10*sig(5)*x[...,isort[0]]**3 + 6*sig(4,1)*x[...,isort[0]]**2*x[...,isort[1]] + sig(3,2)*(3*x[...,isort[0]]*x[...,isort[1]]**2 +  x[...,isort[0]]**3) 
                        + sig(2,2,1)*x[...,isort[2]]*(x[...,isort[0]]**2 + x[...,isort[1]]**2))
                + g[5]*xpow)
    else:
        raise ValueError("n must be between 0 and 5")

def get_all_Dn_new(x, p=0, eps=0.):
    """Get the all the derivatives of a Green's function g
    Using the recurrence relation from Tausch (2003)
    http://dx.doi.org/10.1090/conm/329/05866 (See Section 3)
    Holds for arbitrary radial Green's functions (also non-harmonic cases)

    We want to compute D(n)G(0). The recurrence relates D(n)G(q) to D(n-1)G(q+1) and D(n-2)G(q+1)

    Watch out: sign of x is defined different than previous versions
    """
    gs = _get_gs_v2(x, p+1, eps=eps)

    comb, index_map = define_index_maps(p)

    # Parent relations can be precomputed for the recursive update can be precomputed
    p1s, p2s = np.zeros((len(comb),), dtype=np.int32), np.zeros((len(comb),), dtype=np.int32)
    imaxs = np.zeros((len(comb),), dtype=np.int32)

    w2s = np.zeros((len(comb),), dtype=np.float32)
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
    w1s = jnp.take_along_axis(x, imaxs[None,:], axis=-1).T

    # Now do the dynamical programming update iteratively. We write everything so that
    # we can broadcast over the multipole dimension
    # Dn = jnp.zeros((p_to_ncomb[p],) + x.shape[:-1], dtype=x.dtype).at[0].set(gs[p])
    Dn = gs[p][None,:]
    for q in range(p-1, -1, -1):
        imax = p_to_ncomb[p-q]
        D1 = w1s[1:imax] * Dn[p1s[1:imax]]
        D2 = w2s[1:imax,None] * Dn[p2s[1:imax]]
        Dn = jnp.concatenate((gs[q][None,:], D1 + D2), axis=0)

    return jnp.stack(Dn, axis=-1)
get_all_Dn_new.jit = jax.jit(get_all_Dn_new, static_argnames=("p",))

def potential_direct_sum(x, m=1., n2lim=1e8, eps=1e-5):
    N = x.shape[0]

    nmax = int(np.ceil(n2lim / len(x)))
    nev = int(np.ceil(x.shape[0] / nmax))

    def potential_over_range(i1, i2):
        xi = x[jnp.arange(nmax) + i1]
        # Compute vector distances to all other particles
        r_ij2 = jnp.sum((x - xi[:,None])**2, axis=-1)
        distinv = jnp.where(r_ij2 < 1e-30, 0., 1./jnp.sqrt(r_ij2 + eps**2)) # avoid self-interaction

        return - jnp.sum(m * distinv, axis=1)

    def handle_interval(_, i):
        return None, potential_over_range(nmax * i, nmax * (i + 1))

    _, phis = jax.lax.scan(handle_interval, None, jnp.arange(nev))

    return jnp.concatenate(phis)[0:N]
potential_direct_sum.jit = jax.jit(potential_direct_sum, static_argnames=("n2lim",))

# =============================== Single Interaction Functions =================================== #

def single_multipole_to_local(mp, dx, p=2, eps=0.):
    """Returns the expansion coefficients for the interaction between two nodes"""
    combs, index_of_mp = define_index_maps(p)
    gs = _get_gs(dx, nmax=p, eps=eps)
    D = [_get_Dn(-dx, gs, nx=ks[0], ny=ks[1], nz=ks[2]) for ks in combs]

    Lk = []
    
    for i,ks in enumerate(combs):
        k = np.sum(ks)
        val = 0.
        for ns in multipole_powers(p-k):
            n = np.sum(ns)
            fac = fact[n]/(fact[ns[0]] * fact[ns[1]] * fact[ns[2]])
            
            Dnk = D[index_of_mp[ks[0]+ns[0], ks[1]+ns[1], ks[2]+ns[2]]]
            Qn = mp[...,index_of_mp[ns[0], ns[1], ns[2]]]

            val += Dnk * Qn * fac / fact[n]
        Lk.append(- (-1.)**k / (fact[ks[0]] * fact[ks[1]] * fact[ks[2]])  * val)
        
    return jnp.stack(Lk, axis=-1)

def single_multipole_to_point(mp, dx, p=2, eps=0.):
    """The potential of a multipole expanded at 0 evaluated at dx"""
    combs, index_of_mp = define_index_maps(p)
    gs = _get_gs(dx, nmax=p, eps=eps)

    pot = 0.
    for i,c in enumerate(combs):
        # Number of ways of choosing (c0, c1, c2) given c0+c1+c2 = n, divided by n! (from Taylor expansion):
        fac =  1./(fact[c[0]] * fact[c[1]] * fact[c[2]])
        
        D = _get_Dn(-dx, gs, nx=c[0], ny=c[1], nz=c[2])
        pot = pot - fac * D * mp[...,index_of_mp[c[0], c[1], c[2]]]
    
    return pot

def single_monopole_to_local(mass, dx, p=2, eps=0.):
    """The expansion of a pointmass evaluated at xloc_minus_xmp
    """
    combs = multipole_powers(p)
    gs = _get_gs(dx, nmax=p, eps=eps)
    L = []
    
    for i,ks in enumerate(combs):
        k = np.sum(ks)

        val = mass * _get_Dn(-dx, gs, nx=ks[0], ny=ks[1], nz=ks[2])
        L.append(- (-1.)**k / (fact[ks[0]] * fact[ks[1]] * fact[ks[2]])  * val)
    
    return jnp.stack(L, axis=-1)

# ================================== FFT based interactions ====================================== #

def sym_to_multiindex_grid(M, p=3, pad=False):
    """Converts a flat multi-degree symmetric tensor to a multi-index grid.
    Each grid point (nx,ny,nz) corresponds to an element of Mijklm... 
    where nx indices are 0, ny indices are 1, nz indices are 2, etc.
    """
    nvecs, imap = define_index_maps(p)
    if pad:
        Mgrid = jnp.zeros((M.shape[0], 2*p+2, 2*p+2, 2*p+2), dtype=jnp.float32)
    else:
        Mgrid = jnp.zeros((M.shape[0], p+1, p+1, p+1), dtype=jnp.float32)
    Mgrid = Mgrid.at[:, nvecs[:,0], nvecs[:,1], nvecs[:,2]].set(M)
    return Mgrid

def get_all_Dn(x, p=0, eps=0.):
    """Get the all the derivatives of a Green's function g
    Using the recurrence relation from Tausch (2003)
    http://dx.doi.org/10.1090/conm/329/05866 (See Section 3)
    Holds for arbitrary radial Green's functions (also non-harmonic cases)

    asgrid: if True, returns the derivatives on a grid in multi-index notation
    otherwise returns a flattened array in multipole-order. In this case always limit_nsum=True
    limit_nsum: if True, only returns the derivatives with nx+ny+nz <= p, otherwise returns 
        all derivatives with ni <= p
    """
    gs = _get_gs_v2(x, p+1, eps=eps)

    def get_Dn_gm(nvec, m):
        n = np.sum(nvec)
        if np.min(nvec) < 0:
            return 0.
        elif n == 0:
            return gs[m]
        else:
            ndir = np.argmax(nvec) # Always lower the highest derivative
            evec = 1*(np.arange(3)==ndir)
            val = (nvec[ndir]-1)*get_Dn_gm(tuple(np.array(nvec)-2*evec), m+1) + x[...,ndir] * get_Dn_gm(tuple(nvec - evec), m+1)
            return val
    get_Dn_gm = lru_cache(maxsize=None)(get_Dn_gm) # Lazy dynamic programming, we reuse previous terms
    
    Ds = [get_Dn_gm(tuple(nvec), 0) for nvec in multipole_powers(p)]
    return jnp.stack(Ds, axis=-1)
get_all_Dn.jit = jax.jit(get_all_Dn, static_argnames=("p",))


# def get_Dgrid(dx, p=3, pad=False):
#     assert False, "Deprecated, use get_all_Dn instead"
#     nvecs, imap = define_index_maps(p)
#     gs = _get_gs_v2(np.linalg.norm(dx,axis=-1), p)
#     Ds = jnp.stack([_get_Dn(dx, gs, nvec[0], nvec[1], nvec[2]) for nvec in nvecs], axis=-1)

#     return sym_to_multiindex_grid(Ds, p=p, pad=pad)

# def single_multipole_to_local_via_fft(mp, dx, p=3, eps=0.):
#     assert False, "This function is not working yet, have to put correct combinatorial factors"
#     from math import factorial as fac
#     dx = -dx
#     nvecs, imap = define_index_maps(p)

#     Dgrid = jnp.zeros((len(mp), 2*p+2, 2*p+2, 2*p+2), dtype=dx.dtype)
#     Dgrid = Dgrid.at[...,:p+1, :p+1, :p+1].set(get_all_Dn(dx, p=p, as_grid=True, eps=eps))
#     Dgridk = jnp.fft.rfftn(Dgrid, axes=(-1, -2, -3))
    
#     Mgrid = sym_to_multiindex_grid(mp, p=p, pad=True)
#     Mgridk = jnp.fft.rfftn(Mgrid, axes=(-1, -2, -3))

#     Lgrid = jnp.fft.irfftn(Dgridk * Mgridk, axes=(-1, -2, -3))

#     Lflat = jnp.zeros_like(mp)
#     for i, nvec in enumerate(nvecs):
#         Lflat = Lflat.at[:, i].set(Lgrid[:, nvec[0], nvec[1], nvec[2]] / (-(-1)**np.sum(nvec) *fac(nvec[0]) * fac(nvec[1]) * fac(nvec[2])))
    
#     return Lflat

# =============================== Listed Interaction Functions =================================== #

def ilist_multipole_to_points(mpnodes, xnodes, xpart, ibounds, interactions, imask=None, max_size=100, p=2, eps=0.):
    """
    Directly compute the potential of a multipole at lists of particles
    """
    if imask is None: imask = jnp.ones(len(interactions), dtype=bool)

    ileaf, inode  = interactions.T
    
    iarange = jnp.arange(max_size)

    iparts = ibounds[ileaf,None] + iarange[:]

    nparts = ibounds[ileaf + 1] - ibounds[ileaf]
    ipart_valid = iarange < nparts[:,None]

    weights = single_multipole_to_point(mpnodes[inode][:,None], xpart[iparts] - xnodes[inode][:,None], p=p, eps=eps)

    phi = jnp.zeros((len(xpart)), dtype=jnp.float32)
    phi = phi.at[iparts].add(jnp.where(imask[:,None] & ipart_valid, weights, 0.))
    
    return phi
ilist_multipole_to_points.jit = jax.jit(ilist_multipole_to_points, static_argnames=("max_size", "p", "eps"))

def ilist_monopoles_to_local(xnodes, xpart, mass, ibounds, interactions, imask=None, max_size=100, p=2, eps=0.):
    """
    Directly compute the potential expansion of groups of particles at nodes via interaction lists.
    """
    if imask is None: imask = jnp.ones(len(interactions), dtype=bool)

    inode, ileaf = interactions.T
    
    iarange = jnp.arange(max_size)

    iparts = ibounds[ileaf,None] + iarange[:]

    nparts = ibounds[ileaf + 1] - ibounds[ileaf]
    ipart_valid = iarange < nparts[:,None]

    dx = xnodes[inode][:,None] - xpart[iparts]
    combs = multipole_powers(p)
    gs = _get_gs(dx, nmax=p, eps=eps)
    Ls = []

    mps = mass[iparts]
    
    for i,ks in enumerate(combs):
        val = (mps*ipart_valid) * _get_Dn(-dx, gs, nx=ks[0], ny=ks[1], nz=ks[2])
        Li = - (-1.)**np.sum(ks) / (fact[ks[0]] * fact[ks[1]] * fact[ks[2]])  * val
        Ls.append(jnp.sum(Li, axis=1))
    
    Ls = jnp.stack(Ls, axis=-1)

    loc = jnp.zeros((len(xnodes), Ls.shape[-1]), dtype=jnp.float32)
    loc = loc.at[inode].add(jnp.where(imask[:,None], Ls, 0.))
    
    return loc
ilist_monopoles_to_local.jit = jax.jit(ilist_monopoles_to_local, static_argnames=("max_size", "p", "eps"))

def ilist_monopoles_to_points(pos, mass, ibounds, interactions, imask=None, max_size=100, eps=0.):
    """
    Directly compute the potential via interaction lists.
    ibounds are the splitting indices of different bins in the pos array
    interactions is of shape (n_interactions, 2) and contains pairs of bin-indices
    """
    if imask is None: imask = jnp.ones(len(interactions), dtype=bool)

    i1, i2 = interactions.T
    
    iarange = jnp.arange(max_size)

    phi = jnp.zeros(pos.shape[0], dtype=jnp.float32)

    n1s = ibounds[i1 + 1] - ibounds[i1]
    n2s = ibounds[i2 + 1] - ibounds[i2]

    i1s = ibounds[i1,None,None] + iarange[:,None]
    i2s = ibounds[i2,None,None] + iarange[None,:]
    
    x1s, x2s = pos[i1s], pos[i2s]

    dxmat = x1s - x2s  # Has shape (len(interactions), max_size, max_size, 3)
    rmat2 = jnp.sum(dxmat**2, axis=-1)

    i1valid = iarange[:,None] < n1s[:,None,None]
    i2valid = iarange[None,:] < n2s[:,None,None]
    
    rinv = jnp.where((rmat2 > 0) & i1valid & i2valid & imask[:,None,None], 1. / jnp.sqrt(rmat2 + eps**2), 0.0)
    phi = phi.at[i1s[...,0]].add(jnp.sum(-mass[i2s]*rinv, axis=2))
    
    return phi
ilist_monopoles_to_points.jit = jax.jit(ilist_monopoles_to_points, static_argnames=("max_size", "eps"))

# ==================== Functions for evaluating interaction lists in loops ======================= #

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
    iarange = jnp.arange(chunk_size)
    num_chunks = (iend-istart + chunk_size - 1) // chunk_size # corresponds to ceil((iend-istart) / chunk_size)

    def body(i, y):
        i0 = istart + i * chunk_size
        return f(y, x[i0 + iarange], i0 + iarange < iend)

    return jax.lax.fori_loop(0, num_chunks, body, y0)

def evaluate_ilists_node_node(xnodes, multipoles, interactions, istart, iend, p=2, chunk_fac=4, use_cj=True, eps=0.):
    chunk_size = int(len(xnodes) * chunk_fac)

    if use_cj:
        loc = cj.multipoles.ilist_multipole_to_local(multipoles, xnodes, interactions, iminmax=jnp.array((istart, iend)), p=p, eps=eps)
    else:
        loc = jnp.zeros(multipoles.shape, dtype=jnp.float32)

        def eval_node_node(loc, iab, mask):
            weights = single_multipole_to_local(multipoles[iab[:,1]], xnodes[iab[:,0]] - xnodes[iab[:,1]], p=p, eps=eps)
            weights = jnp.where(mask[:,None], weights, 0.)
            return loc.at[iab[:,0]].add(weights)
        loc = _reduce_fsum_chunked(eval_node_node, loc, interactions, istart, iend, chunk_size=chunk_size)

    return loc
evaluate_ilists_node_node.jit = jax.jit(evaluate_ilists_node_node, static_argnames=("p", "chunk_fac", "eps", "use_cj"))

def evaluate_ilists_leaf_to_node(xnodes, xpart, mpart, leaf_bounds, interactions, istart, iend, p=2, max_leaf_size=64, chunk_fac=1., eps=0.):
    chunk_size = int(len(xnodes) * chunk_fac)

    loc = jnp.zeros(xnodes.shape[:-1] + (p_to_ncomb[p],), dtype=jnp.float32)

    def eval_node_from_leaf(loc, iab, mask):
        return loc + ilist_monopoles_to_local(xnodes, xpart, mpart, leaf_bounds, jnp.abs(iab),
                                              imask=mask, max_size=max_leaf_size, p=p, eps=eps)
    loc = _reduce_fsum_chunked(eval_node_from_leaf, loc, interactions, istart, iend, chunk_size=chunk_size)

    return loc
evaluate_ilists_leaf_to_node.jit = jax.jit(evaluate_ilists_leaf_to_node, static_argnames=("p", "max_leaf_size", "chunk_fac", "eps"))

def evaluate_ilists_node_to_leaf(xnodes, multipoles, xpart, leaf_bounds, interactions, istart, iend, p=2, max_leaf_size=64, chunk_fac=0.4, eps=0.):
    chunk_size = int(len(xpart) * chunk_fac)

    phi = jnp.zeros(xpart.shape[0], dtype=jnp.float32)

    def eval_leaf_node(phi, iab, mask):
        return phi + ilist_multipole_to_points(multipoles, xnodes, xpart, leaf_bounds, jnp.abs(iab), 
                                               imask=mask, max_size=max_leaf_size, p=p, eps=eps)
    phi = _reduce_fsum_chunked(eval_leaf_node, phi, interactions, istart, iend, chunk_size=chunk_size)

    return phi
evaluate_ilists_node_to_leaf.jit = jax.jit(evaluate_ilists_node_to_leaf, static_argnames=("p", "max_leaf_size", "chunk_fac", "eps"))

def evaluate_ilists_leaf_leaf(xpart, mpart, leaf_bounds, interactions, istart, iend, max_leaf_size=64, chunk_fac=4., use_cj=True, eps=0.):
    if use_cj:
        assert eps > 0, "Cannot use custom JAX with eps = 0"
        f, phi = cj.forces.ilist_force(xpart, leaf_bounds, -interactions, iminmax=jnp.array((istart, iend)),
                                       mass=mpart, eps=eps)
    else:
        phi = jnp.zeros(xpart.shape[0], dtype=jnp.float32)
        chunk_size = int(len(leaf_bounds) * chunk_fac)
        def eval_leaf_leaf(phi, iab, mask):
            return phi + ilist_monopoles_to_points(xpart, mpart, leaf_bounds, jnp.abs(iab), imask=mask, max_size=max_leaf_size, eps=eps)
        phi = _reduce_fsum_chunked(eval_leaf_leaf, phi, interactions, istart, iend, chunk_size=chunk_size)

    return phi
evaluate_ilists_leaf_leaf.jit = jax.jit(evaluate_ilists_leaf_leaf, static_argnames=("max_leaf_size", "chunk_fac", "use_cj", "eps"))