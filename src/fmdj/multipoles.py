import jax
import jax.numpy as jnp
import numpy as np
from . import config
from .variants import vm, make_dispatcher, V

try:
    import fmdj_ffi as cj
except ImportError:
    print("No custom JAX found, using fall-back solutions. This may significantly degrade performance.")
    cj = None

# ============================= Some fixed Combinatorical Computations =========================== #

def multi_to_flat(kx, ky, kz):
    p = kx + ky + kz
    npoff = ((p+2)*(p+1)*p // 6)
    npoff += kz*(2*p + 3 - kz)//2 + ky

    return npoff

def generate_combinations(p):
    """Generate unique triples (i, j, k) such that i + j + k = p."""
    combos = []
    for i in range(p + 1):
        for j in range(p + 1 - i):
            k = p - i - j
            combos.append((k, j, i))
    return combos

fact = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=np.int32)
binomial = np.zeros((8, 8), dtype=np.int32)
for n in range(8):
    for k in range(n + 1):
        binomial[n, k] = fact[n] // (fact[k] * fact[n - k])

rank_to_ncomp = np.array([1, 3, 6, 10, 15, 21, 28, 36, 45])
p_to_ncomb = np.array([1, 4, 10, 20, 35, 56, 84, 120, 165])
# ncomp_to_rank = {1: 0, 3: 1, 6: 2, 10: 3, 15: 4, 21: 5, 28: 6, 36: 7}
# ncomb_to_p = {1: 0, 4: 1, 10: 2, 20: 3, 35: 4, 56: 5, 84: 6, 120: 7}
ncomb_to_p = np.zeros(p_to_ncomb[-1] + 1, dtype=np.int32)
ncomb_to_p[p_to_ncomb] = np.arange(len(p_to_ncomb), dtype=np.int32)
ncomp_to_rank = np.zeros(rank_to_ncomp[-1] + 1, dtype=np.int32)
ncomp_to_rank[rank_to_ncomp] = np.arange(len(rank_to_ncomp), dtype=np.int32)

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
                    Lnew = Lnew + bfac.astype(dx.dtype) * L[...,index_of_mp[i,j,k]] * dx[...,0]**(i-c[0]) * dx[...,1]**(j-c[1]) * dx[...,2]**(k-c[2])
        Lout.append(Lnew)

    return jnp.stack(Lout, axis=-1)

def evaluate_local_potential(L, x):
    """Evaluates the function value of the expansion at x"""
    p = ncomb_to_p[L.shape[-1]]
    combs, index_of_mp = define_index_maps(p)

    phi = 0.
    for index, c in enumerate(combs):
        phi +=  L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

    return phi
evaluate_local_potential.jit = jax.jit(evaluate_local_potential)

def evaluate_local_fphi(L, x):
    """Evaluates the function value of the expansion at x"""
    p = ncomb_to_p[L.shape[-1]]
    combs, index_of_mp = define_index_maps(p)

    phi = 0.
    for index, c in enumerate(combs):
        phi +=  L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

    fx, fy, fz = 0., 0., 0.
    for index, c in enumerate(combs):
        if c[0] > 0:
            fx += - L[...,index] * x[...,0]**(c[0]-1) * x[...,1]**c[1] * x[...,2]**c[2] * c[0]
        if c[1] > 0:
            fy += - L[...,index] * x[...,0]**c[0] * x[...,1]**(c[1]-1) * x[...,2]**c[2] * c[1]
        if c[2] > 0:
            fz += - L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**(c[2]-1) * c[2]
    
    fphi = jnp.stack((fx, fy, fz, phi), axis=-1)

    return fphi
evaluate_local_potential.jit = jax.jit(evaluate_local_potential)

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
    if nmax >= 7: gs.append(-155925./r**15)

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
    def sig(val, nx, ny=-1, nz=-1):
        if ny == -1: # only compare nx
            cond = (nsort[0] == nx)
        elif nz == -1:
            cond = (nsort[0] == nx) & (nsort[1] == ny)
        else:
            cond = (nsort[0] == nx) & (nsort[1] == ny) * (nsort[2] == nz)
        return val if cond else 0

    if n == 0:
        return g[0]
    elif n == 1:
        return g[1]*x[...,isort[0]]
    elif n == 2:
        return sig(g[1], 2) + g[2]*xpow
    elif n == 3:
        return g[2]*(sig(3*x[...,isort[0]], 3) + sig(x[...,isort[1]], 2,1)) + g[3]*xpow
    elif n == 4:
        return (g[2]*(sig(3.,4) + sig(1.,2,2)) 
                + g[3]*(sig(6.*x[...,isort[0]]**2,4) + sig(3.*x[...,isort[0]]*x[...,isort[1]],3,1) 
                        + sig((x[...,isort[0]]**2 + x[...,isort[1]]**2),2,2) 
                        + sig(x[...,isort[1]]*x[...,isort[2]],2,1,1))
                + g[4]*xpow)
    elif n == 5:
        return (g[3]*(sig(15.*x[...,isort[0]],5) + sig(3.*x[...,isort[1]],4,1) + sig(3.*x[...,isort[0]],3,2) + sig(x[...,isort[2]], 2,2,1))
                + g[4]*(sig(10.*x[...,isort[0]]**3, 5) + sig(6.*x[...,isort[0]]**2*x[...,isort[1]],4,1) + sig((3*x[...,isort[0]]*x[...,isort[1]]**2 +  x[...,isort[0]]**3),3,2)
                        + sig(x[...,isort[2]]*(x[...,isort[0]]**2 + x[...,isort[1]]**2),2,2,1) + sig(3.*x[...,isort[0]]*x[...,isort[1]]*x[...,isort[2]],3,1,1))
                + g[5]*xpow)
    else:
        raise ValueError("n must be between 0 and 5")

def get_all_Dn_new(x, p=0, eps=0.):
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
        gs = _get_gs_v2(x, p+1, eps=eps)
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
get_all_Dn_new.jit = jax.jit(get_all_Dn_new, static_argnames=("p", "eps"))

def potential_direct_sum(x, m=1., n2lim=1e8, eps=1e-5):
    N = x.shape[0]

    nmax = int(np.ceil(n2lim / len(x)))
    nev = int(np.ceil(x.shape[0] / nmax))

    def potential_over_range(i1, i2):
        xi = x[jnp.arange(nmax, dtype=jnp.int32) + i1]
        # Compute vector distances to all other particles
        r_ij2 = jnp.sum((x - xi[:,None])**2, axis=-1)
        distinv = jnp.where(r_ij2 < 1e-30, 0., 1./jnp.sqrt(r_ij2 + eps**2)) # avoid self-interaction

        return - jnp.sum(m * distinv, axis=1)

    def handle_interval(_, i):
        return None, potential_over_range(nmax * i, nmax * (i + 1))

    _, phis = jax.lax.scan(handle_interval, None, jnp.arange(nev, dtype=jnp.int32))

    return jnp.concatenate(phis)[0:N]
potential_direct_sum.jit = jax.jit(potential_direct_sum, static_argnames=("n2lim",))

# =============================== Single Interaction Functions =================================== #

def single_multipole_to_local(mp, dx, p=2, eps=0.):
    """Returns the expansion coefficients for the interaction between two nodes"""
    combs, index_of_mp = define_index_maps(p)
    D = get_all_Dn_new(-dx, p=p, eps=eps)

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

def single_multipole_to_point(mp, dx, p=2, eps=0.):
    """The potential of a multipole expanded at 0 evaluated at dx"""
    combs, index_of_mp = define_index_maps(p)
    gs = _get_gs(dx, nmax=p, eps=eps)

    pot = 0.
    for i,c in enumerate(combs):
        # Number of ways of choosing (c0, c1, c2) given c0+c1+c2 = n, divided by n! (from Taylor expansion):
        fac =  1./(fact[c[0]] * fact[c[1]] * fact[c[2]])
        
        D = _get_Dn(-dx, gs, nx=c[0], ny=c[1], nz=c[2])
        pot = pot - fac.astype(dx.dtype) * D * mp[...,index_of_mp[c[0], c[1], c[2]]]
    
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
    iarange = jnp.arange(chunk_size, dtype=jnp.int32)
    num_chunks = (iend-istart + chunk_size - 1) // chunk_size # corresponds to ceil((iend-istart) / chunk_size)

    def body(i, y):
        i0 = istart + i * chunk_size
        return f(y, x[i0 + iarange], i0 + iarange < iend)

    return jax.lax.fori_loop(0, num_chunks, body, y0)


def _ilist_node_to_node_base(xnodes, multipoles, interactions, irange, cfg : config.Config):
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

# ================================== Dispatcher Functions ======================================== #
# Below we define the default behaviour of these function
# However, they can be replaced by custom variants through the pattern
# vm.ilist_leaf_to_leaf["mytag"] = Variant(myfunction)
# If the config sets priority to "mytag" the new version will be prefered, e.g.:
# myconfig = config.Config(tags=("mytag", "base"))

ilist_node_to_node = make_dispatcher(vm[V.ilist_node_to_node], _ilist_node_to_node_base, add_jit=True)