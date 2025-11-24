import jax
import jax.numpy as jnp
import numpy as np
from .config import Config
from .data import Multipoles, TreePlane, PosMass

# ============================= Some fixed Combinatorical Computations =========================== #

def iterate_multi_indices(p, istart=0):
    i = 0
    for n in range(p+1):
        for nz in range(n+1):
            for ny in range(n - nz +1):
                nx = n - ny - nz
                if i >= istart:
                    yield i, (nx, ny, nz)
                i += 1

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

def evaluate_local_fphi(L, x):
    """Evaluates the function value of the expansion at x"""
    p = ncomb_to_p[L.shape[-1]]

    phi = 0.
    for index, c in iterate_multi_indices(p):
        phi +=  L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**c[2]

    fx, fy, fz = 0., 0., 0.
    for index, c in iterate_multi_indices(p):
        if c[0] > 0:
            fx += - L[...,index] * x[...,0]**(c[0]-1) * x[...,1]**c[1] * x[...,2]**c[2] * c[0]
        if c[1] > 0:
            fy += - L[...,index] * x[...,0]**c[0] * x[...,1]**(c[1]-1) * x[...,2]**c[2] * c[1]
        if c[2] > 0:
            fz += - L[...,index] * x[...,0]**c[0] * x[...,1]**c[1] * x[...,2]**(c[2]-1) * c[2]
    
    fphi = jnp.stack((fx, fy, fz, phi), axis=-1)

    return fphi
evaluate_local_fphi.jit = jax.jit(evaluate_local_fphi)
