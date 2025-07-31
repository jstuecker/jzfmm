import jax
import jax.numpy as jnp
from jax.experimental import checkify
from .octree import Octree, organize_particles, get_compressed_binary_tree, get_reduced_octree
from .multipoles import calculate_multipoles_for_tree

# ================================ Tree Preperation functions===================================== #

def build_octree_with_multipoles(pos, mass, max_leaf_size=64, p=2):
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
    morton, pos_sorted, isort = organize_particles(pos)
    mass_sorted = jnp.broadcast_to(mass, pos_sorted.shape[0])[isort]

    btree = get_compressed_binary_tree(morton)
    octree = get_reduced_octree(btree, pos_sorted, mass_sorted, max_leaf_size=max_leaf_size)
    octree = calculate_multipoles_for_tree(octree, pos_sorted, mass_sorted, p=p)
    
    return octree, pos_sorted, mass_sorted, isort

# ================================== Some utility methods ======================================== #

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def scatter_masked(y, value, mask, offset=0, get_offsets=False):
    """Emulates y[offset:offset+sum(mask)] = value[mask], but jit-compatible."""
    off, num = offset_sum(mask * 1)
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

    lvl_oct = octree.level_binary // 3
    is_intermediate = jnp.where(octree.level_binary > 0, lvl_oct[octree.parent] == lvl_oct, False)

    # For now we assume L=0 for leaves, since they will be directly summed over...
    # However, in principle the distance calculation for the opening criterio is not 100%
    # right for this case. Possibly we should open all leave-node cases?
    L1 = jnp.where(isleafA, 0., jnp.ldexp(1., -lvl_oct[nodeA])) # 1/2**lvlA
    L2 = jnp.where(isleafB, 0., jnp.ldexp(1., -lvl_oct[nodeB])) # 1/2**lvlB

    x1 = jnp.where(isleafA[:,None], octree.xleaf[-nodeA], octree.xnode[nodeA])
    x2 = jnp.where(isleafB[:,None], octree.xleaf[-nodeB], octree.xnode[nodeB])
    r = jnp.linalg.norm(x1 - x2, axis=-1)

    need_open = (L1 + L2 > thetamax * r) | (nodeA == nodeB)
    imA, imB = is_intermediate[nodeA] & (nodeA > 0), is_intermediate[nodeB] & (nodeB > 0)
    openA = ((need_open & (L1 >= L2) & ~imB) | imA) & (nodeA > 0)
    openB = ((need_open & (L2 >= L1) & ~imA) | imB) & (nodeB > 0)
    return openA, openB

# @partial(jax.jit, static_argnames=('bits', 'ilist_fac', 'clist_fac', 'check_fac', 'maxiter'))
@checkify.checkify
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

    # Expected errors:
    checkify.check(nclist <= clist.shape[0], "Check list overflow. Increase clist_fac")
    checkify.check(nilist <= ilist.shape[0], "Interaction list overflow. Increase ilist_fac")
    # Unexpected errors:
    checkify.check(i < maxiter, "Something went wrong... Not finished after many many iterations")
    checkify.check(nclist == 0, "Something went wrong... Checklist wasn't emptied properly")

    return ilist, nilist