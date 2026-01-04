import numpy as np
import jax
import jax.numpy as jnp

from typing import Tuple
from fmdj_cuda import ffi_tree
from .tools import conditional_callback, div_ceil
from .data import TreePlane, PosMass, PackedArray, TreeHierarchy, InteractionList
from .config import Config, CommunicationConfig, TreeConfig
from .multipoles import center_of_mass
from .tools import cumsum_starting_with_zero, masked_prefix_sum, div_ceil
from .comm import get_rank_info, send_to_left, send_to_right

jax.ffi.register_ffi_target("PosZorderSort", ffi_tree.PosZorderSort(), platform="CUDA")
jax.ffi.register_ffi_target("SummarizeLeaves", ffi_tree.SummarizeLeaves(), platform="CUDA")
jax.ffi.register_ffi_target("FindNodeBoundaries", ffi_tree.FindNodeBoundaries(), platform="CUDA")
jax.ffi.register_ffi_target("GetNodeGeometry", ffi_tree.GetNodeGeometry(), platform="CUDA")
jax.ffi.register_ffi_target("SearchSortedZ", ffi_tree.SearchSortedZ(), platform="CUDA")
jax.ffi.register_ffi_target("GetBoundaryExtendPerLevel", ffi_tree.GetBoundaryExtendPerLevel(), platform="CUDA")


# ------------------------------------------------------------------------------------------------ #
#                                         Helper Functions                                         #
# ------------------------------------------------------------------------------------------------ #

def lvl_to_ext(level_binary):
    olvl, omod = level_binary//3, level_binary % 3
    levels_3d = jnp.stack((olvl, olvl + (omod >= 2).astype(jnp.int32), olvl + (omod >= 1).astype(jnp.int32)),axis=-1)
    return 2.**levels_3d

def get_node_box(x, level_binary):
    node_size = lvl_to_ext(level_binary)
    node_cent = x + jnp.sign(x)*(0.5 * node_size - jnp.mod(jnp.abs(x), node_size))
    return node_cent, node_size

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

def _pos_zorder_sort_impl(x, block_size=64):
    assert x.dtype == jnp.float32
    assert x.shape[-1] == 3

    # To optimize memory layout, we bundle position and id together into a single array
    # We later need to reinterprete the output to extract positions and ids
    out_type = jax.ShapeDtypeStruct((x.shape[0],4), jnp.int32)
    # This is a guess, how much a temporary storage in cub::DeviceMergeSort::SortKeys requires
    # If we estimate too little, an error will be thrown from the C++ code:
    tmp_buff_type = jax.ShapeDtypeStruct((x.shape[0] + np.maximum(1024, x.shape[0]//16), 4), jnp.int32)
    isort = jax.ffi.ffi_call("PosZorderSort", (out_type, tmp_buff_type), vmap_method="sequential")(x, block_size=np.uint64(block_size))[0]

    pos = isort[:, :3].view(jnp.float32)
    ids = isort[:, 3].view(jnp.int32)

    return pos, ids

def pos_zorder_sort(x):
    @jax.custom_vjp
    def eval(x):
        return _pos_zorder_sort_impl(x)
    
    def eval_fwd(x):
        pos, ids = eval(x)
        return (pos, ids), ids
    
    def eval_bwd(ids, g):
        gpos, gids = g
        # Scatter the gradients back to the original ordering
        gpos_unsort = jax.numpy.zeros_like(gpos).at[ids].set(gpos)
        return (gpos_unsort,)
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(x)
pos_zorder_sort.jit = jax.jit(pos_zorder_sort)

def search_sorted_z(xz, xz_query, block_size=64, leaf_search=False):
    """Finds the indices in xz where elements of xz_query would be inserted to keep order.
    This is similar to np.searchsorted, but works for 3D points sorted in Z-order.
    On equality maintains the rule: xz[idx] < v <= xz[idx+1]
    if leaf_search is True, it is assumed that xz contains one point per leaf and we 
    return the index of the leaf that the query point belongs to.
    """
    assert xz.dtype ==  xz_query.dtype == jnp.float32
    assert xz.shape[-1] == xz_query.shape[-1] == 3

    out_type = jax.ShapeDtypeStruct((xz_query.shape[0],), jnp.int32)
    inds = jax.ffi.ffi_call("SearchSortedZ", (out_type,))(
        xz, xz_query, block_size=np.uint64(block_size), leaf_search=leaf_search)[0]
    return inds
search_sorted_z.jit = jax.jit(search_sorted_z, static_argnames=("block_size", "leaf_search"))

def create_coarse_leaves(posz: jnp.ndarray, leaf_size: int = 32, block_size: int = 64, alloc_size: int | None = None) -> jnp.ndarray:
    if alloc_size is None:
        alloc_size = int(div_ceil(len(posz), np.maximum(leaf_size//2, 1))) + 1

    out_type = jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int32)

    nleaf = jnp.ones((posz.shape[0],), dtype=jnp.int32)
    xnleaf = jnp.concatenate((posz, nleaf[:,None].view(jnp.float32)), axis=-1)

    npart = jnp.sum(~jnp.isnan(posz[...,0]))

    flag_split = jax.ffi.ffi_call("SummarizeLeaves", (out_type,), vmap_method="sequential")(
        xnleaf, npart, max_size=np.int32(leaf_size),
        block_size=np.uint64(block_size), scan_size=np.int32(leaf_size+1))[0]

    # Check that the allocation was big enough
    def alloc_err(filled, size):
        raise RuntimeError(f"Coarsen Leaves: allocation too small: filled {filled}, size {size}.\n"
                            "Increase alloc_fac_nodes in FMMConfig.")
    nfilled = jnp.sum(flag_split > -1000)
    flag_split = flag_split + conditional_callback(
        nfilled > alloc_size, alloc_err, nfilled, alloc_size,
    )
    
    splits = jnp.where(flag_split > -1000, size=alloc_size, fill_value=npart)[0]

    return splits
create_coarse_leaves.jit = jax.jit(create_coarse_leaves, static_argnames=("leaf_size", "block_size"))

def determine_znode_boundaries(posz: jnp.ndarray, block_size: int = 64, nleaves: jnp.array = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Builds a Z-order tree from positions"""
    if nleaves is None:
        nleaves = jnp.array(len(posz))
    
    # Define positions that lie beyond the local domain to get first and last node right
    rank, ndev, axis_name = get_rank_info()
    if ndev > 1:
        npart = jnp.sum(~jnp.isnan(posz[...,0]))
        pos_left = send_to_right(posz[npart-1], axis_name, invalid_val=jnp.nan)
        pos_right = send_to_left(posz[0], axis_name, invalid_val=jnp.nan)
        pos_bound = jnp.stack([pos_left, pos_right], axis=0)
    else:
        pos_bound = jnp.full((2,3), jnp.nan, dtype=posz.dtype)

    out_types = (jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int32),)*3
    lvl, lbound, rbound = jax.ffi.ffi_call("FindNodeBoundaries", out_types)(
        posz, pos_bound, nleaves, block_size=np.uint64(block_size)
    )

    return lvl, lbound, rbound
determine_znode_boundaries.jit = jax.jit(determine_znode_boundaries)

def get_node_geometry(posz: jnp.ndarray, lbound: jnp.ndarray, rbound: jnp.ndarray, 
                      num: jnp.array = None, block_size: int = 64
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if num is None:
        num = jnp.array(len(lbound))

    out_types = (jax.ShapeDtypeStruct((lbound.shape[0],), jnp.int32),
                 jax.ShapeDtypeStruct((lbound.shape[0], 3), jnp.float32),
                 jax.ShapeDtypeStruct((lbound.shape[0], 3), jnp.float32))
    
    lvl, node_cent, node_ext = jax.ffi.ffi_call("GetNodeGeometry", out_types)(
        posz, lbound, rbound, num, block_size=np.uint64(block_size)
    )

    return lvl, node_cent, node_ext
get_node_geometry.jit = jax.jit(get_node_geometry)

def distr_boundary_extend(posz, npart=None, block_size: int = 64):
    rank, ndev, axis_name = get_rank_info()

    if npart is None:
        npart = jnp.sum(~jnp.isnan(posz[...,0]))
    
    nlevels = 388 + 450 + 1
    out_types = (jax.ShapeDtypeStruct((nlevels,), jnp.int32),)

    irange = jnp.array([0, npart])

    # Distance from the left boundary where each levels node ends
    xleft = send_to_right(posz[npart-1], axis_name, invalid_val=-jnp.inf)
    ext_lr = jax.ffi.ffi_call("GetBoundaryExtendPerLevel", out_types)(
        xleft, irange, posz, block_size=np.uint64(block_size), left=True
    )[0]

    # Distance from the right boundary where each levels node starts
    xright = send_to_left(posz[0], axis_name, invalid_val=jnp.inf)
    ext_rl = jax.ffi.ffi_call("GetBoundaryExtendPerLevel", out_types)(
        xright, irange, posz, block_size=np.uint64(block_size), left=False
    )[0]

    ext_rr = send_to_left(ext_lr, axis_name, invalid_val=0)
    ext_ll =  send_to_right(ext_rl-npart, axis_name, invalid_val=0)

    return ext_ll, ext_lr, ext_rl, ext_rr

# ------------------------------------------------------------------------------------------------ #
#                                       Domain Decomposition                                       #
# ------------------------------------------------------------------------------------------------ #

def determine_npart(pos):
    """Determines the number of valid particles (that are not nan)"""
    valid = ~jnp.isnan(pos)
    return jnp.sum(valid[...,0] & valid[...,1] & valid[...,2])

def distributed_zsort(pos: jnp.ndarray, cfg_com: CommunicationConfig):
    axis_name = cfg_com.axis_name

    rank = jax.lax.axis_index(axis_name=axis_name)
    ndev = jax.lax.axis_size(axis_name=axis_name)

    if ndev == 1:
        return pos_zorder_sort(pos)[0]

    npart = determine_npart(pos)
    nparttot = jax.lax.psum(npart, axis_name=axis_name)

    # Sample based domain decomposition
    nsamp = cfg_com.zsort_domain_samples
    key = jax.random.key(0)
    isamp = jax.random.randint(key, shape=(nsamp,), minval=0, maxval=npart)
    posall = jax.lax.all_gather(pos[isamp], axis_name=axis_name, tiled=True)
    xpivot = pos_zorder_sort(posall)[0][nsamp::nsamp]
    xpivot = jnp.pad(xpivot, ((1,1), (0,0)), constant_values=jnp.inf).at[0].set(-jnp.inf)

    # Now organize and determine which chunks need to be send to each rank
    posz, idz = pos_zorder_sort(pos)
    spl = search_sorted_z(posz, xpivot)

    def err(n1, n2):
        raise MemoryError(f"Size of buffer is too small: need {n1}, have {n2}.\n")
    nneed = jnp.max(spl[1:] - spl[:-1])
    spl = spl + conditional_callback(nneed > pos.shape[0], err, nneed, pos.shape[0])

    from .comm import all_to_all_with_splits, global_splits

    pos = all_to_all_with_splits(posz, spl, jnp.full_like(posz, jnp.nan), axis_name=axis_name)
    posz, idz = pos_zorder_sort(pos)

    # We have posz globally and locally in z-order now
    # Let's do another communication step to improve the balance
    npart = determine_npart(posz)
    spl_have = global_splits(npart, axis_name=axis_name)
    spl_target = (jnp.arange(0, ndev+1) * (nparttot // ndev)).at[-1].set(nparttot)
    spl_send = jnp.clip(spl_target - spl_have[rank], 0, npart)

    posz = all_to_all_with_splits(posz, spl_send, jnp.full_like(posz, jnp.nan), axis_name=axis_name)

    return posz

def shift_particles_left(pos, nsend, max_send, npart=None):
    rank, ndev, axis_name = get_rank_info()

    if npart is None:
        npart = jnp.sum(~jnp.isnan(pos[...,0]))
    
    # Validate that send buffer is large enough
    def send_size_err(nsend, max_send):
        raise ValueError(f"Cannot fit {nsend} particles into buffer of size {max_send}!")
    npart = npart + conditional_callback(
        nsend >= max_send, send_size_err, nsend, max_send,
    )

    # Validate that particle array has enough free space
    nget = send_to_left(nsend, axis_name, invalid_val=0)
    def array_size_err(nhave, nget, nsend, nmax):
        raise MemoryError(
            f"Cannot shift particles: have={nhave}, get={nget}, send={nsend}, max={nmax}."
            "(fix: larger allocaction)"
        )
    npart = npart + conditional_callback((
        npart + nget - nsend >= len(pos)), array_size_err, npart, nget, nsend, len(pos)
    )

    # Send the particles
    pos_get = send_to_left(pos[0:max_send], axis_name, invalid_val=jnp.nan)

    # Delete the particles that were send
    iar = jnp.arange(len(pos))
    pos = jnp.where((iar >= nsend)[:,None], pos, jnp.nan)
    pos = jnp.roll(pos, -nsend, axis=0)

    # Insert the received particles
    idx = jnp.arange(max_send)
    idx = jnp.where(idx < nget, npart - nsend + idx, len(pos)) # discard indices beyond nadd
    pos = pos.at[idx].set(pos_get)

    return pos, npart + nget - nsend

def adjust_domain_for_nodesize(posz, max_node_size, npart=None):
    """Shifts particles so that nodes with size <= max_node_size always lie on a single GPU"""
    if npart is None:
        npart = jnp.sum(~jnp.isnan(posz[...,0]))

    ext_ll, ext_lr, ext_rl, ext_rr = distr_boundary_extend(posz, npart=npart)
    npart_l = ext_lr - ext_ll
    
    ilvl_max = jnp.argmax(jnp.where(npart_l <= max_node_size, npart_l, 0))

    npshift = ext_lr[ilvl_max]

    posz, npart = shift_particles_left(posz, npshift, max_send=max_node_size, npart=npart)

    return posz, npart

# ------------------------------------------------------------------------------------------------ #
#                                      Tree Building Functions                                     #
# ------------------------------------------------------------------------------------------------ #

def estimate_node_number(npart, max_node_size, alloc_fac=1.):
    """Gives an estimate of the number of nodes
    
    Assumes that particles are grouped in z-order into nodes that have <= node_size particles each.
    """
    # Since most nodes with size <= max_node_size/2 can be grouped, we generally get a reasonably 
    # safe estimate by dividing node_size/2. However, the estimate is not totally guaranteed, since
    # imbalanced distributions may require some nodes with size >= max_node_size/2 may block
    # multiple nodes with size <= max_node_size/2 from being summarized.
    return int(div_ceil(npart*alloc_fac, np.maximum(max_node_size//2, 1))) + 1

def define_tree_level_node_sizes(npart: int, cfg_tree: TreeConfig):
    max_num_leaves = estimate_node_number(npart, cfg_tree.max_leaf_size)

    nlevels = np.log(max_num_leaves / cfg_tree.stop_coarsen) / np.log(cfg_tree.coarse_fac)
    nlevels = np.maximum(int(np.ceil(nlevels)), 1)

    node_sizes = [int(cfg_tree.max_leaf_size) * (cfg_tree.coarse_fac ** i) for i in range(0, nlevels)]

    return node_sizes

def define_split_hierarchy(posz: jnp.ndarray, node_sizes: Tuple[int], alloc_size: int
                           ) -> Tuple[jnp.ndarray, PackedArray, PackedArray]:
    """Finds the splitting point of the tree hierarchy

    returns
      ispl: the splitting points of leafs in the particle array
      ispl_n2n: a PackedArray that defines the node to node splitting points per level
      ispl_n2l: a PackedArray that defines the node to leaf splitting points per level
    """
    nlevels = len(node_sizes)
    
    ispl =  create_coarse_leaves(posz, leaf_size=node_sizes[0], alloc_size=alloc_size)

    nleaves = jnp.argmax(ispl)
    lvl, lbound, rbound = determine_znode_boundaries(posz[ispl[:-1]], nleaves=nleaves)

    npart_node = ispl[rbound] - ispl[lbound]

    # At each level of the hierarchy, the active nodes are determined by the node sizes
    active_on_level = npart_node[None,:] > jnp.array(node_sizes, dtype=jnp.int32)[:,None]
    # Calculate a prefix accross hierarchy levels to densly stack the nodes later
    offsets = jnp.cumsum(active_on_level.flatten()).reshape(active_on_level.shape)
    level_spl = jnp.pad(offsets[:,-1], (1,0), constant_values=0) # save level start/end points
    nnodes_on_level = level_spl[1:] - level_spl[:-1] - 1 # -1 since splits are always 1 larger than nodes

    # Check that the allocation is big enough
    def alloc_err(filled, size):
        raise RuntimeError(f"Tree allocation too small: filled {filled}, size {size}.\n"
                            "Increase alloc_fac_nodes in Config")
    level_spl = level_spl + conditional_callback(
        level_spl[-1] > alloc_size, alloc_err, level_spl[-1], alloc_size,
    )

    # correct offsets to exclude the element at hand and invalidate inactive elements:
    offsets = jnp.where(active_on_level, offsets - active_on_level, alloc_size)
    
    # node-to-leaf relation is given by the leaf that is active at the node location
    ispl_n2l = jnp.zeros(alloc_size, dtype=jnp.int32).at[offsets].set(offsets[0:1,:])
    ispl_n2l = PackedArray(ispl_n2l, level_spl, fill_values=nnodes_on_level[0])

    # node-to-node relation is given by the last level node that is active at the node location
    # for the leaf-level we insert the leaf to particle relation here
    ilevel = jnp.arange(nlevels)
    value = jnp.where(ilevel[:,None] == 0, ispl, offsets[ilevel-1,:] - level_spl[ilevel-1,None])
    ispl_n2n = jnp.zeros(alloc_size, dtype=jnp.int32).at[offsets].set(value)
    # out of bounds access shall give nnodes of next smaller level (or npart for leaves):
    fill_val = jnp.pad(nnodes_on_level[:-1], (1,0), constant_values=ispl[-1])
    ispl_n2n = PackedArray(ispl_n2n, level_spl, fill_values=fill_val)
    
    return ispl, ispl_n2l, ispl_n2n

def get_tree_mass_centers(part: PosMass, ispl_n2n: PackedArray) -> Tuple[PackedArray, PackedArray]:
    prop_array_spl = cumsum_starting_with_zero(ispl_n2n.ispl[1:] - ispl_n2n.ispl[:-1] - 1)

    def handle_mcent_level(i, carry):
        node_mcent, node_mass, posm = carry
        posm = center_of_mass(ispl_n2n.get(i, size=len(posm.mass)+1), posm)
        node_mass = node_mass.set(i, posm.mass, num=ispl_n2n.num(i)-1)
        node_mcent = node_mcent.set(i, posm.pos, num=ispl_n2n.num(i)-1)
        return node_mcent, node_mass, posm
    
    posm = center_of_mass(ispl_n2n.get(0), part)
    npos = PackedArray(posm.pos, ispl=prop_array_spl, fill_values=jnp.nan)
    node_mass = PackedArray(posm.mass, ispl=prop_array_spl, fill_values=jnp.nan)

    node_mcent, node_mass, _ = jax.lax.fori_loop(
        1, ispl_n2n.nlevels(), handle_mcent_level, (npos, node_mass, posm)
    )

    return node_mcent, node_mass

def build_tree_hierarchy(part: PosMass | jnp.ndarray, cfg_tree: TreeConfig) -> TreeHierarchy:
    """Builds a tree hierarchy from z-order positions

    The zeroth level of the tree corresponds to leaves, which contain multiple particles.
    Nodes (and leaves) are selected so that they are as big as possible while not containing more
    than a maximum number of particles that starts at cfg_tree.max_leaf_size and increases per level
    by a factor cfg_tree.coarse_fac.
    
    Nodes are parameterized through a set of splits. For example the particles that lie in the leaf
    with index i are given py part[ispl_n2n.get(level=0)[i]: ispl_n2n.get(level=0)[i+1]]
    The ith node of level n contain all level n-1 nodes in the range :
    ispl_n2n.get(n)[i]: ispl_n2n.get(n)[i+1]

    In jax memory size needs to be known at compile time, but the required number of nodes is 
    data dependent on each level. To limit the number of allocations that we need to predict, we
    use the PackedArray class, that helps us to stack multiple different levels into a single
    continguous array, but to access it "almost" as if they were separate arrays.
    """

    if isinstance(part, jnp.ndarray):
        assert part.shape[-1] == 3
        posz = part
    elif hasattr(part, "pos"):
        posz = part.pos
    else:
        raise ValueError("Invalid input particles")

    node_sizes = define_tree_level_node_sizes(len(posz), cfg_tree)
    nlevels = len(node_sizes)

    alloc_size = estimate_node_number(len(posz), cfg_tree.max_leaf_size, cfg_tree.alloc_fac_nodes)

    ispl, ispl_n2l, ispl_n2n = define_split_hierarchy(posz, node_sizes, alloc_size)

    # We can handle all levels at once for node geometry:
    ispl_n2p = ispl[ispl_n2l.data] # node to particle relation
    lvl, geom_cent, ext = get_node_geometry(
        posz, ispl_n2p[:-1], ispl_n2p[1:], num=ispl_n2l.nfilled()-1
    )
    # However, the splits are discontinuous at level boundaries. We have to delete the extra entries
    lvl = jnp.delete(lvl, ispl_n2l.ispl[1:-1]-1, assume_unique_indices=True)
    geom_cent = jnp.delete(geom_cent, ispl_n2l.ispl[1:-1]-1, axis=0, assume_unique_indices=True)

    # node property arrays are on each level one element smaller than the splitting point arrays
    prop_array_spl = cumsum_starting_with_zero(ispl_n2n.ispl[1:] - ispl_n2n.ispl[:-1] - 1)
    lvl = PackedArray(lvl, prop_array_spl, fill_values=-1000)
    geom_cent = PackedArray(geom_cent, prop_array_spl, fill_values=jnp.nan)

    if cfg_tree.mass_centered:
        assert hasattr(part, "mass"), "To use mass centering, please provide PosMass input"
        nmass_cent, nmass = get_tree_mass_centers(part, ispl_n2n)
    else:
        nmass_cent, nmass = None, None
        
    # Predict maximum plane (at compile time) and check whether the prediction was large enough
    # Note: This step can probably be skipped after I adapted the code to use fixed size arrays
    plane_sizes = [estimate_node_number(len(posz), node_sizes[i], alloc_fac=cfg_tree.alloc_fac_nodes)
                   for i in range(0, nlevels)]

    def plane_size_err(num, sizes):
        raise RuntimeError(f"Tree allocation too small: \nplanes filled: {num} \nsize: {sizes}.\n"
                            "Increase alloc_fac_nodes in FMMConfig.")
    
    nsizes = ispl_n2n.ispl[1:] - ispl_n2n.ispl[:-1] - 1
    ispl_n2l.ispl = ispl_n2l.ispl + conditional_callback(
        jnp.any(nsizes > jnp.array(plane_sizes)), plane_size_err,
        nsizes, jnp.array(plane_sizes),
    )

    th = TreeHierarchy(
        ispl_n2n, ispl_n2l, lvl, geom_cent, mass = nmass, mass_cent = nmass_cent,
        plane_sizes = plane_sizes
    )
    
    return th
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg_tree'])

# ------------------------------------------------------------------------------------------------ #
#                                     Interaction List Helpers                                     #
# ------------------------------------------------------------------------------------------------ #

def dense_interaction_list(size: int, nnodes: jnp.ndarray = None) -> InteractionList:
    """A dense interaction list where all nodes interact with all other nodes.

    size: size of the node array that will use the interaction list. (Required at compile time)
    nnodes: actual number of filled nodes (Can be dynamic, used to invalidating unused nodes)
    """

    if nnodes is None: # size = nnodes will only work outside of jit
        nnodes = jnp.array(size, dtype=jnp.int32)  
    dtype = nnodes.dtype

    nfilled = nnodes*nnodes

    idx = jnp.arange(size*size)
    ilist = idx % nnodes
    
    ispl = jnp.minimum(jnp.arange(0, size+1, dtype=dtype) * nnodes, nfilled)
    
    return InteractionList(ispl=ispl, iother=ilist, nfilled=nfilled)
dense_interaction_list.jit = jax.jit(dense_interaction_list, static_argnames=['size'])

def grouped_dense_interaction_list(nnodes: jnp.ndarray | int, size_ilist: int,
                                   size_super: int | None = None, ngroup: int = 32
                                   ) -> Tuple[jnp.ndarray, InteractionList, jnp.ndarray]:
    nsuper_nodes = div_ceil(nnodes, ngroup)
    ninteractions = nsuper_nodes*nsuper_nodes

    # Somehow if I keep the following check, I get slight differences in the resulting gradients
    # Have to check carefully whether I ever read from any uninitialized memory, which might
    # get affected by reordering the computation through this callback (?)

    def ilist_size_error(n, size):
        raise MemoryError(f"Cannot fit {n}*{n} interactions into ilist with size {size}")
    nsuper_nodes = nsuper_nodes + conditional_callback(
        ninteractions > size_ilist, ilist_size_error, nsuper_nodes, size_ilist
    )
    
    idx = jnp.arange(size_ilist)
    ilist = jnp.where(idx < ninteractions, idx % nsuper_nodes, 0)
    
    # define the super node to node relation
    if size_super is None:
        size_super = np.ceil(np.sqrt(size_ilist)).astype(np.int64)
    spl_super = jnp.minimum(jnp.arange(size_super+1) * ngroup, nnodes)
    ispl = jnp.minimum(jnp.arange(size_super+1) * nsuper_nodes, ninteractions)

    ilist = InteractionList(ispl=ispl, iother=ilist, nfilled=ninteractions)

    return spl_super, ilist, nsuper_nodes
grouped_dense_interaction_list.jit = jax.jit(
    grouped_dense_interaction_list, static_argnames=["size_ilist", "size_super"]
)