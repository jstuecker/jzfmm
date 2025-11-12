import jax
import jax.numpy as jnp
from .octree import Octree, sort_and_build_octree
from . import multipoles
from . import config
from .variants import Variant, vm, TAG_BASE
from .tools import conditional_callback

# ================================ Tree Preperation functions===================================== #

def build_octree_with_multipoles(pos, mass, max_leaf_size=64, p=2, use_cj=True):
    """Build an octree with multipoles for the given positions and masses.
    
    This function organizes the particles, builds a binary tree, reduces it to an octree,
    and calculates the multipoles for each node in the octree.
    
    Args:
        pos: Particle positions.
        mass: Particle masses.
        max_leaf_size: Maximum number of particles per leaf node.
        p: Order of multipole moments to calculate.
    
    Returns: (octree, pos_sorted, mass_sorted, isort)
    """

    octree, posz, massz, isortz = sort_and_build_octree(pos, mass, use_cj=use_cj, max_leaf_size=max_leaf_size)

    octree = multipoles.calculate_multipoles_for_tree(octree, posz, massz, p=p)
    
    return octree, posz, massz, isortz
build_octree_with_multipoles.jit = jax.jit(build_octree_with_multipoles, 
                                           static_argnames=("max_leaf_size", "p", "use_cj"))

# ================================== Some utility methods ======================================== #

def cumsum_starting_with_zero(x):
    return jnp.pad(jnp.cumsum(x), (1, 0))

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def scatter_masked(y, value, mask, offset=0, get_offsets=False):
    """Emulates y[offset:offset+sum(mask)] = value[mask], but jit-compatible."""
    off, num = offset_sum(mask.astype(jnp.int32))
    off_masked = jnp.where(mask & (offset+off >= 0), offset+off, y.shape[0])

    ynew = y.at[off_masked].set(value)
    if get_offsets:
        return ynew, num + offset, off + offset
    else:
        return ynew, num + offset

# ==================================== Interaction List ========================================== #

def opening_criterion(octree : Octree, nodeA, nodeB, thetamax=0.75):
    """The opening criterion for two nodes, with special handling for leaves=particles
    we need to open all intermediate nodes and all non-well separated nodes (but never leaves)
    if the levels are asymmetric, we only open the larger node
    """
    isleafA, isleafB = nodeA <= 0, nodeB <= 0

    lvl_oct = (octree.level_binary + 2) // 3
    is_intermediate = jnp.where(octree.level_binary <= octree.max_level, 
                                lvl_oct[octree.parent] == lvl_oct, False)

    # For now we assume L=0 for leaves, since they will be directly summed over...
    # However, in principle the distance calculation for the opening criterio is not 100%
    # right for this case. Possibly we should open all leave-node cases?
    one = jnp.array(1., dtype=octree.xnode.dtype)
    L1 = jnp.where(isleafA, 0., jnp.ldexp(one, lvl_oct[nodeA])) # 2**lvlA
    L2 = jnp.where(isleafB, 0., jnp.ldexp(one, lvl_oct[nodeB])) # 2**lvlB

    x1 = jnp.where(isleafA[:,None], octree.xleaf[-nodeA], octree.xnode[nodeA])
    x2 = jnp.where(isleafB[:,None], octree.xleaf[-nodeB], octree.xnode[nodeB])
    r = jnp.linalg.norm(x1 - x2, axis=-1)

    need_open = (L1 + L2 > thetamax * r) | (nodeA == nodeB)
    imA, imB = is_intermediate[nodeA] & (nodeA > 0), is_intermediate[nodeB] & (nodeB > 0)
    openA = ((need_open & (L1 >= L2) & ~imB) | imA) & (nodeA > 0)
    openB = ((need_open & (L2 >= L1) & ~imA) | imB) & (nodeB > 0)
    return openA, openB

def build_interaction_list(octree : Octree, thetamax=0.75, ilist_fac=512, clist_fac=128, check_fac=8):
    """Build an interaction list for the octree, using a binary tree walk
    
    This is done by keeping track of a check list of node pairs that need to be checked. For each
    checked pair we either add the interaction to the interaction list or we add the children to the
    check list.

    These lists may become very large (~ a few hundred times the number of nodes) and we need to
    allocate a buffer that is large enough to hold them in advance. Therefore, to avoid memory
    issues, it is key that we have much fewer nodes than particles (max_leaf_size >> 1)

    thetamax: Opening angle
    ilist_fac: Allocation factor for the interaction list (in units of nleaves)
    clist_fac: Allocation factor for the check list (in units of nleaves)
    check_fac: How many interactions to check in each iteration (in units of nleaves)

    returns: err, (ilist, nilist) 
       where err is a possible error message that indicates e.g. if we ran out of memory
    """
    maxiter = 100000 # If we go beyond this, for sure something went wrong...

    max_leaves = len(octree.level_binary) - 1
    ilist = jnp.empty((int(max_leaves*ilist_fac), 2), dtype=jnp.int32)
    clist = jnp.empty((int(max_leaves*clist_fac), 2), dtype=jnp.int32)

    n_per_step = int(max_leaves * check_fac)

    def step(carry):
        i, clist, ilist, nclist, nilist = carry

        # Pop the last n_per_step interactions from the check list
        istart = jnp.clip(nclist - n_per_step, 0, None)
        isel = istart + jnp.arange(n_per_step, dtype=jnp.int32)
        interaction, valid = clist[isel], (isel < nclist)
        nclist = istart
        nodeA, nodeB = interaction.T
        valid = valid & ((nodeA <= 0) | octree.is_valid[nodeA])
        valid = valid & ((nodeB <= 0) | octree.is_valid[nodeB])

        openA, openB = opening_criterion(octree, nodeA, nodeB, thetamax=thetamax)

        # Where there is no opening, we simply add the interaction and are done with this pair
        ilist, nilist = scatter_masked(ilist, interaction, valid & ~openA & ~openB, offset=nilist)

        # Handle non well separated nodes
        def add_check(node1, node2, mask, offset):
            return scatter_masked(clist, jnp.stack((node1, node2), axis=-1), mask, offset=offset)
        
        childAl, childAr = octree.lchild[nodeA], octree.rchild[nodeA]
        childBl, childBr = octree.lchild[nodeB], octree.rchild[nodeB]

        # Case open both
        clist, nclist = add_check(childAl, childBl, valid & openA & openB, nclist)
        clist, nclist = add_check(childAl, childBr, valid & openA & openB, nclist)
        clist, nclist = add_check(childAr, childBl, valid & openA & openB, nclist)
        clist, nclist = add_check(childAr, childBr, valid & openA & openB, nclist)

        # Case open only A
        clist, nclist = add_check(childAl, nodeB, valid & openA & ~openB, nclist)
        clist, nclist = add_check(childAr, nodeB, valid & openA & ~openB, nclist)

        # Case open only B
        clist, nclist = add_check(nodeA, childBl, valid & ~openA & openB, nclist)
        clist, nclist = add_check(nodeA, childBr, valid & ~openA & openB, nclist)
        
        return i+1, clist, ilist, nclist, nilist
    
    def while_condition(carry):
        i, clist, ilist, nclist, nilist = carry
        return (nclist > 0) & (nilist < ilist.shape[0]) & (nclist < clist.shape[0]) & (i < maxiter)
    
    # Start with root node interaction
    clist = clist.at[0].set(jnp.array([octree.rchild[0], octree.rchild[0]], dtype=jnp.int32))  

    i, clist, ilist, nclist, nilist = jax.lax.while_loop(
        while_condition, step,
        (0, clist, ilist, 1, 0)
    )

    # Expected errors
    def cerror(n1, n2): 
        raise MemoryError(f"The check list is too small (need: {n1} have: {n2})")
    nilist = nilist + conditional_callback(nclist>clist.shape[0], cerror, nclist, clist.shape[0])

    def ierror(n1, n2): 
        raise MemoryError(f"The interaction list is too small (need: {n1} have: {n2})")
    nilist = nilist + conditional_callback(nilist>ilist.shape[0], ierror, nilist, ilist.shape[0])

    # Unexpected errors
    def uerror(i_max): 
        raise RuntimeError(f"Something went wrong... Not finished after {i_max} iterations")
    nilist = nilist + conditional_callback(i>=maxiter, uerror, maxiter)

    return ilist, nilist
build_interaction_list.jit = jax.jit(
    build_interaction_list, static_argnames=("ilist_fac", "clist_fac", "check_fac"))

def organize_interactions(interaction_list, nfilled, sort=False):
    """Organizes the interactions. """
    inodeA, inodeB = interaction_list.T

    # Interaction types: # 0: node-node, 1: node-leaf, 2: leaf-node, 3: leaf-leaf 4: invalid
    itype = (2*(inodeA <= 0).astype(jnp.int32) + (inodeB <= 0).astype(jnp.int32) 
             + 4*(jnp.arange(len(interaction_list), dtype=jnp.int32) >= nfilled).astype(jnp.int32))

    if sort:
        # Sort interactions by type and receiving nodes inodeA
        interactions = interaction_list[jnp.lexsort((inodeB, inodeA, itype))]

        iend = jnp.array([jnp.sum(itype <= i) for i in (0,1,2,3,8)])
        isplits = jnp.concatenate((jnp.array([0]), iend))
    else:
        # Only sort by type. Since we have only 4 types, we can do this manually with prefix sums
        offsets = 0
        isplits = [0]
        for i in range(0, 4):
            offs, num = offset_sum((itype == i).astype(jnp.int32))
            offsets = jnp.where(itype == i, isplits[-1] + offs, offsets)
            isplits.append(isplits[-1] + num)
        offsets = jnp.where(itype > 3, len(interaction_list), offsets)

        interactions = interaction_list.at[offsets].set(interaction_list)

    return interactions, jnp.array(isplits)
organize_interactions.jit = jax.jit(organize_interactions, static_argnames=("sort",))

# ============================= Evaluate Interaction Lists ======================================= #

def evaluate_interaction_lists(octree : Octree, posz, massz, ilist, nilist, cfg : config.Config, sort=False):
    ilist, iranges = organize_interactions(ilist, nilist, sort=sort)
    
    Loc = multipoles.ilist_node_to_node(
        octree.xnode, octree.mp, ilist, iranges[0:2], cfg=cfg)
    Loc = Loc + multipoles.ilist_leaf_to_node(
        octree.xnode, posz, massz, octree.leaf_particle_bounds, ilist, iranges[1:3], cfg=cfg)
    Loc = multipoles.local_to_local_via_height(octree, Loc)

    phi = multipoles.evaluate_local_potential(Loc[octree.node_of_particle], 
                                    posz - octree.xnode[octree.node_of_particle])
    phi = phi + multipoles.ilist_node_to_leaf(
        octree.xnode, octree.mp, posz, octree.leaf_particle_bounds, ilist, iranges[2:4], cfg=cfg)
    phi = phi + multipoles.ilist_leaf_to_leaf(
        posz, massz, octree.leaf_particle_bounds, ilist, iranges[3:5], cfg=cfg)
    
    return phi
evaluate_interaction_lists.jit = jax.jit(evaluate_interaction_lists, static_argnames=("cfg", "sort"))

# =================================== Master functions =========================================== #

#eps=0., thetamax=0.75, max_leaf_size=64, p=2, use_cj=True, 
def fast_multipole_potential(pos, mass, cfg : config.Config, return_sorted=False):
    """Fast Multipole Method for calculating potentials and forces.
    
    This function builds an octree with multipoles, constructs an interaction list,
    and evaluates the interaction lists to compute the potential and forces.
    
    Args:
        pos: Particle positions.
        mass: Particle masses.
        thetamax: Opening angle for the FMM.
        max_leaf_size: Maximum number of particles per leaf node.
        p: Order of multipole moments to calculate.
        use_cj: Whether to use custom JAX for performance optimizations.
    
    Returns: (phi, force)
    """
    octree, posz, massz, isortz = build_octree_with_multipoles.jit(
        pos, mass, max_leaf_size=cfg.fmm.max_leaf_size, p=cfg.fmm.p)
    
    ilist, nilist = build_interaction_list.jit(octree, thetamax=cfg.fmm.opening_angle)
    # ilist, iranges = organize_interactions.jit(ilist, nilist, sort=False)
    
    phiz = evaluate_interaction_lists.jit(octree, posz, massz, ilist, nilist, cfg=cfg)

    if return_sorted:
        return posz, massz, isortz, phiz
    else:
        return jnp.zeros_like(phiz).at[isortz].set(phiz)
fast_multipole_potential.jit = jax.jit(fast_multipole_potential,
    static_argnames=("cfg", "return_sorted"))


# ------------------------------------------------------------------------------------------------ #
#                                              New FMM                                             #
# ------------------------------------------------------------------------------------------------ #

def new_fmm_potential(pos, mass, cfg : config.Config, return_sorted=False):
    import custom_jax as cj
    import custom_jax.cj_new_tree as cnt
    import fmdj.new_tree as nt

    if mass is None:
        mass = jnp.ones((pos.shape[0],), dtype=pos.dtype)
    elif jnp.shape(mass) != jnp.shape(pos)[:-1]:
        mass = jnp.broadcast_to(mass, pos.shape[:-1])

    posz, isortz = cj.tree.pos_zorder_sort(pos)
    particlesz = nt.Particles(pos=posz, mass=mass[isortz])

    th = nt.build_tree_hierarchy(particlesz, cfg)
    loc, ilist = nt.evaluate_interaction_hierarchy(th, cfg=cfg)

    parent = th[0].icoarse_of_fine()
    phi_loc = multipoles.evaluate_local_potential(loc[parent], posz - th[0].mp.center()[parent])

    fphi = cnt.cj_new_force_and_pot(particlesz, th[0], ilist, cfg=cfg)
    
    phiz = fphi[:,3] + phi_loc

    if return_sorted:
        return particlesz.pos, particlesz.mass, isortz, phiz
    else:
        return jnp.zeros_like(phiz).at[isortz].set(phiz)
new_fmm_potential.jit = jax.jit(new_fmm_potential, static_argnames=("cfg", "return_sorted"))

def new_fmm_fphi(pos, mass, cfg : config.Config, return_sorted=False):
    import custom_jax as cj
    import custom_jax.cj_new_tree as cnt
    import fmdj.new_tree as nt

    if mass is None:
        mass = jnp.ones((pos.shape[0],), dtype=pos.dtype)
    elif jnp.shape(mass) != jnp.shape(pos)[:-1]:
        mass = jnp.broadcast_to(mass, pos.shape[:-1])

    posz, isortz = cj.tree.pos_zorder_sort(pos)
    particlesz = nt.Particles(pos=posz, mass=mass[isortz])

    th = nt.build_tree_hierarchy(particlesz, cfg)
    loc, ilist = nt.evaluate_interaction_hierarchy(th, cfg=cfg)

    parent = th[0].icoarse_of_fine()
    fphi_loc = multipoles.evaluate_local_fphi(loc[parent], particlesz.pos - th[0].mp.center()[parent])

    fphi = cnt.cj_new_force_and_pot(particlesz, th[0], ilist, cfg=cfg) + fphi_loc

    if return_sorted:
        return particlesz.pos, particlesz.mass, isortz, fphi
    else:
        fphi_unsorted = jnp.zeros_like(fphi).at[isortz].set(fphi)
        return fphi_unsorted
new_fmm_fphi.jit = jax.jit(new_fmm_fphi, static_argnames=("cfg", "return_sorted"))

def get_force_and_potential(pos, mass, cfg : config.Config, separately=False):
    if cfg.fmm is None: # Use direct summation
        import custom_jax as cj
        fphi = cj.forces.force_and_potential(
            pos, mass, softening=cfg.softening, kahan=True) * cfg.G()
    else:
        fphi = new_fmm_fphi(pos, mass, cfg=cfg) * cfg.G()

    if separately:
        return fphi[:,0:3], fphi[:,3]
    else:
        return fphi
get_force_and_potential.jit = jax.jit(get_force_and_potential, static_argnames=("cfg", "separately"))