import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import checkify
import jax.numpy as jnp
from dataclasses import dataclass, fields

if jax.__version__ <= "0.4.2":
    raise ImportError("This code requires JAX version 0.4.2 or higher. " \
                      "However, we could define a custom decorator for this case")

# ======================================= Tree class ============================================= #

@partial(jax.tree_util.register_dataclass, 
         data_fields=["lbound", "rbound", "lchild", "rchild", "level_binary"], 
         meta_fields=[])
@dataclass
class BinaryTree:
    # Jax arrays
    lbound: jnp.ndarray = None
    rbound: jnp.ndarray = None
    lchild: jnp.ndarray = None
    rchild: jnp.ndarray = None

    level_binary: jnp.ndarray = None

@partial(jax.tree_util.register_dataclass, 
         data_fields=["lchild", "rchild", "is_valid", "level_binary", "xnode", 
                      "xleaf", "mp", "leaf_particle_bounds"], 
         meta_fields=["p"])
@dataclass
class Octree:
    lchild: jnp.ndarray = None
    rchild: jnp.ndarray = None
    is_valid: jnp.ndarray = None

    level_binary: jnp.ndarray = None

    xnode: jnp.ndarray = None
    xleaf: jnp.ndarray = None

    mp: jnp.ndarray = None
    leaf_particle_bounds: jnp.ndarray = None

    p: int = 0

# ===================================== Morton sorting =========================================== #

def pos_to_icoord(pos, L=1., bits=30):
    """Converts position in (-L/2, L/2) to integer grid coordinates"""
    valid = jnp.all((pos[...,0:3] >= -L/2.) & (pos[...,0:3] < L), axis=-1)
    # Adding 2**30 corresponds to L/2 shift:
    # We do this in integers (and not in floats) to not lose precision at x=0
    ipos = jnp.floor(pos[...,0:3] / L * 2.0**bits).astype(jnp.int32) + 2**(bits-1)  
    
    return ipos, valid

def expand_10b(v):
    """
    Expands 10-bit integers into 30-bit by inserting two 0 bits between 
    each input bit.

    Strongly inspired by:
    https://stackoverflow.com/questions/18529057/
    """
    v = v & 0x3ff
    v = (v | v << 16) & 0x30000ff
    v = (v | v << 8) & 0x300f00f
    v = (v | v << 4) & 0x30c30c3
    v = (v | v << 2) & 0x9249249

    return v

def morton_diff_level(morton1, morton2):
    """
    Gets the level of a cell that would include both morton1 and morton2.
    (corresponding to the first different bit, counted from the left.)
    0 corresponds to the largest possible cell and 89 corresponds to the smallest resolved cell.
    90 corresponds to 0 difference in the morton code.
    """
    morton_xor = morton1 ^ morton2

    largest_bit = jnp.floor(jnp.log2(jnp.clip(2*morton_xor,1,None))).astype(jnp.int32)

    comb_largest_bit = jnp.where(largest_bit[...,0] > 0, largest_bit[...,0], 0)
    comb_largest_bit = jnp.where(largest_bit[...,1] > 0, largest_bit[...,1] + 30, comb_largest_bit)
    comb_largest_bit = jnp.where(largest_bit[...,2] > 0, largest_bit[...,2] + 60, comb_largest_bit)

    return 90 - comb_largest_bit

def morton_3int32_to_3int32(ipos):
    """
    converts integer coordinates (range 0...2**30-1) to 3 bit-interleaved morton keys
    the primary key is the last one (to align with jax.lexsort)
    """
    ix, iy, iz = ipos[...,0], ipos[...,1], ipos[...,2]
    i1 = (expand_10b(ix      ) << 2) | (expand_10b(iy      ) << 1) | expand_10b(iz      )
    i2 = (expand_10b(ix >> 10) << 2) | (expand_10b(iy >> 10) << 1) | expand_10b(iz >> 10)
    i3 = (expand_10b(ix >> 20) << 2) | (expand_10b(iy >> 20) << 1) | expand_10b(iz >> 20)

    return jnp.stack([i1, i2, i3], axis=-1)

def organize_particles(pos, return_sorted=True):
    ipos, valid = pos_to_icoord(pos)
    morton = morton_3int32_to_3int32(ipos)
    isort = jnp.lexsort(morton.T)
    if return_sorted:
        return morton[isort], pos[isort], isort
    else:
        return morton, isort

# ====================================== Tree building =========================================== #

def get_parent_binary(levels, lbound, rbound):
    """Decides which of the two bounding nodes is the parent of the node"""
    nnodes = levels.shape[0]
    is_left = ((levels[lbound] >= levels[rbound]) & (lbound > 0)) | (rbound >= nnodes-1)
    return jnp.where(is_left, lbound, rbound), is_left

def find_previous_and_next_lower(lvls):
    """Given a 1d array of integers, find the previous and next lower integer for each element.
    if lvls corresponds to the levels of a binary tree, this gives the extend of each node
    """
    nnodes = len(lvls)
    iarange = jnp.arange(nnodes)
    iprev = jnp.clip(iarange - 1,0,None)
    inext = jnp.clip(iarange + 1,0,nnodes-1)

    def loop_body(args):
        iprev, inext, done = args

        is_lower_prev = (lvls[iprev] < lvls) | (iprev == 0)
        iprev = jnp.where(is_lower_prev, iprev, iprev[iprev])

        is_lower_next = (lvls[inext] < lvls) | (inext == len(lvls)-1)
        inext = jnp.where(is_lower_next, inext, inext[inext])

        done = is_lower_prev & is_lower_next

        return iprev, inext, done
    
    def loop_cond(args):
        iprev, inext, done = args
        return jnp.any(~done)
    
    iprev, inext, done = jax.lax.while_loop(loop_cond, loop_body, 
                                            (iprev, inext, jnp.zeros(nnodes, dtype=bool)))
    
    return iprev, inext

def determine_children(lvls, lbound, rbound):
    """Determines the children by assigning the index through the children"""
    iparent, is_left = get_parent_binary(jnp.array(lvls), lbound, rbound)

    nnodes = len(lvls)
    iarange = jnp.arange(nnodes)

    lchild = -jnp.arange(nnodes) + 1
    rchild = -jnp.arange(nnodes)

    rchild = rchild.at[jnp.where(is_left, iparent, nnodes)].set(iarange)
    lchild = lchild.at[jnp.where(~is_left, iparent, nnodes)].set(iarange)

    # Handle our two fake nodes at the boundary:
    # We can always find the rootnode as rchild[0]
    rootnode = jnp.argmin(lvls[1:-1]) + 1
    lchild = lchild.at[0].set(0).at[-1].set(rootnode)
    rchild = rchild.at[-1].set(nnodes-1).at[0].set(rootnode)

    return lchild, rchild

def get_compressed_binary_tree(morton, version=1) -> BinaryTree:
    max_level = 90

    levels = morton_diff_level(morton[1:], morton[:-1])
    # We put fake levels -1 at the start and end so that all nodes have well defined boundaries
    levels = jnp.concatenate((jnp.array((-1,)), levels,  jnp.array((-1,))))
    nnodes = levels.shape[0]

    iarange = jnp.arange(levels.shape[0])

    last_lower = jnp.zeros_like(levels, dtype=jnp.int32) # Left Parent
    last_lowest_above = jnp.zeros_like(levels, dtype=jnp.int32) # Left Child

    next_lower = jnp.full_like(levels, nnodes-1, dtype=jnp.int32) # Right Parent
    next_lowest_above = jnp.full_like(levels, nnodes-1, dtype=jnp.int32) # Right Child

    tree = BinaryTree()
    tree.level_binary = levels

    if version == 1:
        def iteration(lvl, args):
            last_lower, last_lowest_above, next_lower, next_lowest_above = args
            last_seen = jax.lax.cummax(iarange * (lvl == levels)) # last seen occurance of the value "lvl"

            # last occurence that is lower than the current level
            last_lower = jnp.where((lvl < levels) & (last_seen > last_lower), last_seen, last_lower)

            # the location of the lowest value between last_lower and the current value
            mask = (last_seen > last_lower) & (lvl > levels) & (last_lowest_above <= 0)
            last_lowest_above = jnp.where(mask, last_seen, last_lowest_above)
            
            # Now the same, but other way around
            next_seen = jax.lax.cummin(iarange * (lvl == levels) + nnodes * (lvl != levels), reverse=True)

            next_lower = jnp.where((lvl < levels) & (next_seen < next_lower), next_seen, next_lower)

            mask = (next_seen < next_lower) & (lvl > levels) & (next_lowest_above >= nnodes-1)
            next_lowest_above = jnp.where(mask, next_seen, next_lowest_above)

            return last_lower, last_lowest_above, next_lower, next_lowest_above
        
        last_lower, last_lowest_above, next_lower, next_lowest_above = jax.lax.fori_loop(
            0, max_level+1, iteration, (last_lower, last_lowest_above, next_lower, next_lowest_above))

        # Now interprete these in a tree structure
        # A nodes extend is between the last and next lower refinement level
        tree.lbound = last_lower
        tree.rbound = next_lower

        # The children are the lowest elements between our node and ilbound and irbound
        # If there are no child-nodes, then the child corresponds to a leaf (=particle here)
        # In that case we use a negative index that indicates the location in the leaf array
        tree.lchild = jnp.where(last_lowest_above > 0, 
                                last_lowest_above, -(iarange-1)).at[0].set(0)
        tree.rchild = jnp.where(next_lowest_above < nnodes-1, 
                                next_lowest_above, -iarange).at[-1].set(nnodes-1)
    else:
        tree.lbound, tree.rbound = find_previous_and_next_lower(levels)
        tree.lchild, tree.rchild = determine_children(levels, tree.lbound, tree.rbound)

    return tree