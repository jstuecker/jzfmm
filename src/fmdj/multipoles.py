import jax
import jax.numpy as jnp
from .octree import Octree, get_oct_level_info
import numpy as np

# ============================= Some fixed Combinatorical Computations =============================

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

# =============================== Multipole to Multipole  Operators ================================

def save_divide(a, b):
    return jnp.where(b != 0, a / b, 0.)

def com_via_levels(octree : Octree, pos, mass):
    """Computes the mass and the center of mass of each node in the octree"""
    max_nodes = len(octree.lchild)

    # First add particles into their parent nodes
    m = jnp.zeros(max_nodes, dtype=jnp.float32
                  ).at[octree.node_of_particle].add(mass)
    mx = jnp.zeros((max_nodes, 3), dtype=jnp.float32
                   ).at[octree.node_of_particle].add(pos * mass[:,None])

    # Next propagate information up the tree
    def handle_level(i, carry):
        level_parent = -i
        m, mx = carry

        sel = (octree.level_binary[octree.parent] == level_parent) & octree.is_valid
        ipar = jnp.where(sel, octree.parent, max_nodes)
        
        m = m.at[ipar].add(m)
        mx = mx.at[ipar].add(mx)

        return m, mx
    
    m, mx = jax.lax.fori_loop(-90, 1, handle_level, (m, mx))

    return m, save_divide(mx, m[:,None])


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

def multipoles_via_levels(octree : Octree, pos, mass, p=2, xcom=None):
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

    # To save some time, we may skip the calculation for intermediate levels
    # intermediate levels do not represent cubes, but rather rectangular intermediate splits
    # and we anyways always open them in the tree walk
    level_oct, parent_oct, is_intermediate = get_oct_level_info(octree)
    
    def handle_level(i, mp):
        sel = (level_oct[parent_oct] == -i) & octree.is_valid

        if xcom is not None:
            mpnew = shift_multipoles(mp, x0[parent_oct] - x0, p=p)
        else:
            mpnew = mp

        iparent = jnp.where(sel, parent_oct, max_nodes)
        return mp.at[iparent].add(mpnew)
    
    mp = jax.lax.fori_loop(-jnp.max(level_oct)+1, 1, handle_level, mp)

    return mp