import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import checkify
import jax.numpy as jnp
from dataclasses import dataclass, fields
import numpy as np

if jax.__version__ <= "0.4.2":
    raise ImportError("This code requires JAX version 0.4.2 or higher. " \
                      "However, we could define a custom decorator for this case")

try:
    import fmdj_ffi as cj
except ImportError:
    print("No custom JAX found, using fall-back solutions. This may significantly degrade performance.")
    cj = None

# ======================================= Tree class ============================================= #

@partial(jax.tree_util.register_dataclass, 
         data_fields=["lbound", "rbound", "lchild", "rchild", "level_binary"], 
         meta_fields=["min_level", "max_level"])
@dataclass
class BinaryTree:
    # Jax arrays
    lbound: jnp.ndarray = None
    rbound: jnp.ndarray = None
    lchild: jnp.ndarray = None
    rchild: jnp.ndarray = None
    level_binary: jnp.ndarray = None

    # Meta fields
    # Nodes with particles at identical positions have this level (-(127+23)*3):
    min_level: int = -450 
    # Nodes with particles with different signs have this level (128+1)*3
    max_level: int =  387

@partial(jax.tree_util.register_dataclass, 
         data_fields=["parent", "lchild", "rchild", "level_binary", "is_valid", "height",  
                      "maxheight", "nnodes",
                      "leaf_particle_bounds", "node_of_leaf", "node_of_particle", "xnode", "xleaf", "mp",], 
         meta_fields=["max_leaf_size", "p", "min_level", "max_level"])
@dataclass
class Octree:
    parent: jnp.ndarray = None
    lchild: jnp.ndarray = None
    rchild: jnp.ndarray = None
    level_binary: jnp.ndarray = None
    is_valid: jnp.ndarray = None
    height: jnp.ndarray = None
    maxheight: jnp.ndarray = None
    nnodes: jnp.ndarray = None

    leaf_particle_bounds: jnp.ndarray = None
    node_of_leaf: jnp.ndarray = None
    node_of_particle: jnp.ndarray = None

    xnode: jnp.ndarray = None
    xleaf: jnp.ndarray = None
    mp: jnp.ndarray = None

    max_leaf_size: int = 1
    p: int = 0

    min_level: int = -450 
    max_level: int =  387

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
    isort = jnp.lexsort(morton.T).astype(jnp.int32)
    if return_sorted:
        return morton[isort], pos[isort], isort
    else:
        return morton, isort
organize_particles.jit = jax.jit(organize_particles, static_argnames=("return_sorted",))

# ================================== Binary Tree building ======================================== #

def get_parent_binary(levels, lbound, rbound):
    """Decides which of the two bounding nodes is the parent of the node"""
    nnodes = levels.shape[0]
    is_left = ((levels[lbound] <= levels[rbound]) & (lbound > 0)) | (rbound >= nnodes-1)
    return jnp.where(is_left, lbound, rbound), is_left

def find_previous_and_next_higher(lvls):
    """Given a 1d array of integers, find the previous and next higher integer for each element.
    if lvls corresponds to the levels of a binary tree, this gives the extend of each node
    """
    nnodes = len(lvls)
    iarange = jnp.arange(nnodes, dtype=jnp.int32)
    iprev = jnp.clip(iarange - 1,0,None)
    inext = jnp.clip(iarange + 1,0,nnodes-1)

    def loop_body(args):
        iprev, inext, done = args

        is_higher_prev = (lvls[iprev] > lvls) | (iprev == 0)
        iprev = jnp.where(is_higher_prev, iprev, iprev[iprev])

        is_higher_next = (lvls[inext] > lvls) | (inext == len(lvls)-1)
        inext = jnp.where(is_higher_next, inext, inext[inext])

        done = is_higher_prev & is_higher_next

        return iprev, inext, done
    
    def loop_cond(args):
        iprev, inext, done = args
        return jnp.any(~done)
    
    iprev, inext, done = jax.lax.while_loop(loop_cond, loop_body, 
                                            (iprev, inext, jnp.zeros(nnodes, dtype=bool)))
    
    return iprev, inext
find_previous_and_next_higher.jit = jax.jit(find_previous_and_next_higher)

def determine_children(lvls, lbound, rbound):
    """Determines the children by assigning the index through the children"""
    iparent, is_left = get_parent_binary(jnp.array(lvls), lbound, rbound)

    nnodes = len(lvls)
    iarange = jnp.arange(nnodes, dtype=jnp.int32)

    lchild = -jnp.arange(nnodes, dtype=jnp.int32) + 1
    rchild = -jnp.arange(nnodes, dtype=jnp.int32)

    rchild = rchild.at[jnp.where(is_left, iparent, nnodes)].set(iarange)
    lchild = lchild.at[jnp.where(~is_left, iparent, nnodes)].set(iarange)

    # Handle our two fake nodes at the boundary:
    # We can always find the rootnode as rchild[0]
    rootnode = jnp.argmax(lvls[1:-1]).astype(jnp.int32) + 1
    lchild = lchild.at[0].set(0).at[-1].set(rootnode)
    rchild = rchild.at[-1].set(nnodes-1).at[0].set(rootnode)

    return lchild, rchild
determine_children.jit = jax.jit(determine_children)

def get_compressed_binary_tree(morton) -> BinaryTree:
    tree = BinaryTree(min_level=-90, max_level=0)
    tree.level_binary = -morton_diff_level(morton[1:], morton[:-1])
    # We put fake levels -1 at the start and end so that all nodes have well defined boundaries
    tree.level_binary = jnp.concatenate((jnp.array((tree.max_level+1,), dtype=jnp.int32), tree.level_binary,  
                                         jnp.array((tree.max_level+1,), dtype=jnp.int32)))

    tree.lbound, tree.rbound = find_previous_and_next_higher(tree.level_binary)
    tree.lchild, tree.rchild = determine_children(tree.level_binary, tree.lbound, tree.rbound)

    return tree
get_compressed_binary_tree.jit = jax.jit(get_compressed_binary_tree)

def cj_build_ztree(pos_zsorted : jnp.ndarray) -> BinaryTree:
    """Builds a binary tree from zsorted-positions of particles
    
    returns (tree, morton, isort)
    """
    assert cj is not None

    tree_info = cj.tree.build_ztree(pos_zsorted)

    tree = BinaryTree(level_binary=tree_info[0], lbound=tree_info[1], rbound=tree_info[2],
                      lchild=tree_info[3], rchild=tree_info[4], min_level=-450, max_level=387)

    return tree
cj_build_ztree.jit = jax.jit(cj_build_ztree)

def get_tree_height(tree : Octree | BinaryTree) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Maximum number of children that can be passed to reach a leaf
    
    returns : (height, max_height)
    """
    if isinstance(tree, Octree):
        nnodes = tree.nnodes
    else:
        nnodes = len(tree.level_binary)

    # Performance considerations: In a CUDA program with level sorted nodes, it would be possible
    # to restrict the nodes that need be updated to a smaller range.

    # The loop assumes that no circles exist in the tree in the range [1:nnodes-1]
    # If a circle exists, the loop will never terminate
    def loop_body(carry):
        hmax, height = carry
        height_left = jnp.where(tree.lchild > 0, height[tree.lchild], 0)
        height_right = jnp.where((tree.rchild > 0) & (tree.rchild < nnodes-1), height[tree.rchild], 0)
        height = jnp.maximum(height_left, height_right) + 1
        return hmax+1, height
    def loop_cond(carry):
        hmax, height = carry
        return jnp.max(height) == hmax 
    
    height = jnp.zeros_like(tree.level_binary)
    iterend, height = jax.lax.while_loop(loop_cond, loop_body, (0, height))
    
    # Get the maximum height of the tree:
    # -1 because we end at one iteration higher
    # -1 because of the fake nodes that have the root node as child:
    max_height = iterend - 2 

    return height, max_height
get_tree_height.jit = jax.jit(get_tree_height)

# ===================================== Tree Reduction =========================================== #

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def define_reduced_tree_masks(tree : BinaryTree, max_leaf_size=4):
    nodesize = tree.rbound - tree.lbound
    parent = get_parent_binary(tree.level_binary, tree.lbound, tree.rbound)[0]

    # We keep all valid nodes whose parents exceed the max_leaf_size
    keep = nodesize[parent] > max_leaf_size
    # For nodes that we keep, we can identify leaves as those that are smaller than max_leaf_size
    keep_as_leaf = keep & (nodesize <= max_leaf_size) & (tree.level_binary > tree.min_level)
    # Additionally we may need to keep as leaves nodes that are at identical positions, 
    # even if they have more than max_leaf_size particles, since we cannot split them further
    # Since they all share a parent, we have to make sure to keep only the one of them
    # that the parent knows about
    inode = jnp.arange(len(tree.level_binary), dtype=jnp.int32)
    is_a_child = (tree.lchild[tree.rbound] == inode) | (tree.rchild[tree.lbound] == inode)
    keep_as_leaf = keep_as_leaf | (keep & (tree.level_binary == tree.min_level) & is_a_child)
    
    keep_as_node = keep & (tree.level_binary > tree.min_level) & ~keep_as_leaf

    # Beyond that, we also need to transport some leaves directly from the old to the new tree.
    # These will generally be single particles
    keep_lchild_leaf = keep_as_node & (tree.lchild <= 0) & (tree.level_binary <= tree.max_level)
    keep_rchild_leaf = keep_as_node & (tree.rchild <= 0) & (tree.level_binary <= tree.max_level)

    return keep_as_node, keep_as_leaf, keep_lchild_leaf, keep_rchild_leaf

def define_leaf_maps(tree, keep_as_node, keep_as_leaf, keep_lchild_leaf, keep_rchild_leaf, max_new_leaves):
    max_old_leaves = len(keep_as_node) - 1
    max_old_nodes = len(keep_as_node)

    parent = get_parent_binary(tree.level_binary, tree.lbound, tree.rbound)[0]

    # Figure out where leaves must be put in the new leaf-array
    imap_leaf, num_new_leaves = offset_sum(keep_as_leaf.astype(jnp.int32) 
                                           + keep_lchild_leaf.astype(jnp.int32) 
                                           + keep_rchild_leaf.astype(jnp.int32))

    def masked_update(ar, mask, value, right=False):
        ind = imap_leaf if not right else imap_leaf + keep_lchild_leaf.astype(jnp.int32)
        return ar.at[jnp.where(mask, ind, max_new_leaves+1)].set(value, indices_are_sorted=True)

    # Create pointers to the old parents of new leaves
    parent_of_leaf = jnp.full(max_new_leaves, fill_value=max_old_leaves, dtype=jnp.int32)
    parent_of_leaf = masked_update(parent_of_leaf, keep_as_leaf, parent)
    parent_of_leaf = masked_update(parent_of_leaf, keep_lchild_leaf, jnp.arange(max_old_nodes, dtype=jnp.int32))
    parent_of_leaf = masked_update(parent_of_leaf, keep_rchild_leaf, jnp.arange(max_old_nodes, dtype=jnp.int32), 
                                   right=True)

    # Keep track of the first particle that belongs to each new leaf
    # we make the array one larger, so that the extend of each leaf is always given by i[1:]-i[:-1]
    lbound_leaf = jnp.full(max_new_leaves+1, fill_value=max_old_leaves, dtype=jnp.int32) 
    lbound_leaf = masked_update(lbound_leaf, keep_as_leaf, tree.lbound)
    lbound_leaf = masked_update(lbound_leaf, keep_lchild_leaf, -tree.lchild)
    lbound_leaf = masked_update(lbound_leaf, keep_rchild_leaf, -tree.rchild, right=True)

    # Determine which new leaf an old particle belongs to
    part_leaf_starting_points = jnp.zeros(max_old_leaves, dtype=jnp.int32).at[lbound_leaf].set(1)
    leaf_of_part = jnp.cumsum(part_leaf_starting_points) - 1

    return (imap_leaf, parent_of_leaf, lbound_leaf, leaf_of_part)

def get_new_leaf_positions(leaf_of_part, xpart, mpart, max_new_leaves):
    """Determines the center of mass of new summarized leaf particles."""
    m_in_leaf = jnp.zeros(max_new_leaves, dtype=xpart.dtype
                          ).at[leaf_of_part].add(mpart, indices_are_sorted=True)
    mx_in_leaf = jnp.zeros((max_new_leaves,3), dtype=xpart.dtype
                           ).at[leaf_of_part].add(xpart*mpart[:,None], indices_are_sorted=True)
    x_leaf = mx_in_leaf / m_in_leaf[:,None]

    return x_leaf

def get_reduced_octree(tree : BinaryTree, xpart, mpart, max_leaf_size=4) -> Octree:
    """Defines a reduced octree, where all internal nodes have n > max_leaf_size and 
    all leaves have n <= max_leaf_size. (This rule may be violated for leaves at the highest level,
    where particles below have identical morton keys and can't be split)
    We summarize leaves as pseudo-particles, represented through the center of mass of the 
    particles in the leaf.
    """
    # Something that creates noteworthy complexity in this function is that new leaves can be
    # (1) nodes in the original tree
    # (2) left leaves of the original tree
    # (3) right leaves of the original tree

    max_old_leaves = len(tree.level_binary) - 1
    max_old_nodes = len(tree.level_binary)

    assert max_old_leaves > max_leaf_size, "Please use smaller leaves or more particles"

    # A worst case estimate of the number of new leaves
    max_new_leaves = int(np.ceil(max_old_leaves /  np.clip((max_leaf_size//2),1, None)))
    max_new_nodes = max_new_leaves + 1

    # Determine which nodes and leaves we keep in the new tree
    (keep_as_node, keep_as_leaf, keep_lchild_leaf, keep_rchild_leaf
     ) = define_reduced_tree_masks(tree, max_leaf_size=max_leaf_size)

    # Define maps that relate new and old leave positions
    (imap_leaf, parent_of_leaf, lbound_leaf, leaf_of_part
     ) = define_leaf_maps(tree, keep_as_node, keep_as_leaf, keep_lchild_leaf, keep_rchild_leaf, max_new_leaves)

    # Find out where to put new internal nodes
    imap_node, nnodes = offset_sum(keep_as_node.astype(jnp.int32))
    imap_node = jnp.where(keep_as_node, imap_node, max_new_nodes)
    # With the inverse map it is easier to construct the new tree
    ifrom_node = jnp.zeros(max_new_nodes, dtype=jnp.int32
                           ).at[imap_node].set(jnp.arange(max_old_nodes, dtype=jnp.int32), indices_are_sorted=True)

    def map_node(id): # Returns the node index in the new tree of an old node id
        inew = jnp.where(id <= 0, -leaf_of_part[-id], max_new_leaves)
        inew = jnp.where((id > 0) & keep_as_node[id], imap_node[id], inew)
        inew = jnp.where((id > 0) & keep_as_leaf[id], -imap_leaf[id], inew)
        return inew

    newtree = Octree(nnodes=nnodes, max_leaf_size=max_leaf_size, 
                     min_level=tree.min_level, max_level=tree.max_level)
    inewnode = jnp.arange(max_new_nodes, dtype=jnp.int32)

    iparent = get_parent_binary(tree.level_binary, tree.lbound, tree.rbound)[0]
    newtree.parent = map_node(iparent[ifrom_node])
    newtree.lchild = map_node(tree.lchild[ifrom_node])
    newtree.rchild = map_node(tree.rchild[ifrom_node])

    newtree.level_binary = tree.level_binary[ifrom_node]
    newtree.is_valid = (inewnode > 0) & (inewnode < nnodes-1)
    newtree.height, newtree.maxheight = get_tree_height(newtree)

    newtree.leaf_particle_bounds = lbound_leaf
    newtree.node_of_leaf = imap_node[parent_of_leaf]
    newtree.node_of_particle = imap_node[parent_of_leaf][leaf_of_part]
    
    newtree.xleaf = get_new_leaf_positions(leaf_of_part, xpart, mpart, max_new_leaves)
    
    return newtree
get_reduced_octree.jit = jax.jit(get_reduced_octree, static_argnames=("max_leaf_size",))

def put_nodes_in_level_order(octree : Octree) -> Octree:
    """Sorts nodes so that all nodes of the same level are next to each other."""
    print("Warning, I should update this method to sort by height rather than level!")

    max_nodes = len(octree.level_binary)
    inode = jnp.arange(len(octree.level_binary), dtype=jnp.int32)
    # lvels, but modified so that we keep the last node and invalid nodes at the end when sorting
    level = octree.level_binary.at[jnp.where(inode >= octree.nnodes-1, inode, max_nodes)].set(100)

    # For a given new node index, which original index it came from:
    isort = jnp.lexsort((inode, -level)).astype(jnp.int32) # With lexsort nodes of the same level keep their rel. order
    # For a given original index, which new index it is at:
    inv_i = jnp.empty_like(isort).at[isort].set(jnp.arange(isort.size, dtype=jnp.int32))

    # Children may be leaves -- only map their indices if they are nodes
    lchild = jnp.where(octree.lchild > 0, inv_i[octree.lchild], octree.lchild)
    rchild = jnp.where(octree.rchild > 0, inv_i[octree.rchild], octree.rchild)

    assert octree.xnode is None, "Have not considered xnode sorting here"
    assert octree.mp is None, "Have not considered mp sorting here"

    newtree = Octree(


        parent=inv_i[octree.parent[isort]],
        lchild=lchild[isort],
        rchild=rchild[isort],
        level_binary=octree.level_binary[isort],
        is_valid=octree.is_valid[isort],
        height=octree.height[isort],
        maxheight=octree.maxheight,
        nnodes=octree.nnodes,
        
        leaf_particle_bounds=octree.leaf_particle_bounds,
        node_of_leaf=inv_i[octree.node_of_leaf],
        node_of_particle=inv_i[octree.node_of_particle],

        xleaf = octree.xleaf,

        max_leaf_size=octree.max_leaf_size, 
        p=octree.p,
        min_level=octree.min_level,
        max_level=octree.max_level
    )
    
    return newtree, isort, inv_i
put_nodes_in_level_order.jit = jax.jit(put_nodes_in_level_order)

def sort_and_build_octree(pos0, mass0, use_cj=True, max_leaf_size=64) -> Octree:
    if use_cj:
        posz, isort_z = cj.tree.pos_zorder_sort(pos0)
        btree_z = cj_build_ztree(posz)
        massz = mass0[isort_z]
        octree = get_reduced_octree(btree_z, posz, massz, max_leaf_size=max_leaf_size)
        return octree, posz, massz, isort_z
    else:
        if use_cj:
            print("Warning: customjax not found, using fall-back solution.")
        morton, posz, isortz = organize_particles(pos0)
        btree = get_compressed_binary_tree(morton)
        massz = mass0[isortz]
        octree = get_reduced_octree(btree, posz, massz, max_leaf_size=max_leaf_size)
        return octree, posz, mass0, isortz
sort_and_build_octree.jit = jax.jit(sort_and_build_octree, 
                                    static_argnames=("use_cj", "max_leaf_size"))

# ================================== Dispatcher Functions ======================================== #

