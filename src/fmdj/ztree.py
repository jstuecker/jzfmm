import numpy as np
import jax
import jax.numpy as jnp

from typing import Tuple
from fmdj_cuda import ffi_tree
from .tools import conditional_callback, div_ceil
from .data import TreePlane, PosMass, TreeHierarchy, PackedArray, NewTreeHierarchy
from .config import Config
from .multipoles import center_of_mass
from .tools import cumsum_starting_with_zero

jax.ffi.register_ffi_target("PosZorderSort", ffi_tree.PosZorderSort(), platform="CUDA")
jax.ffi.register_ffi_target("SummarizeLeaves", ffi_tree.SummarizeLeaves(), platform="CUDA")
jax.ffi.register_ffi_target("FindNodeBoundaries", ffi_tree.FindNodeBoundaries(), platform="CUDA")
jax.ffi.register_ffi_target("GetNodeGeometry", ffi_tree.GetNodeGeometry(), platform="CUDA")

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

def create_coarse_leaves(posz: jnp.ndarray, leaf_size: int = 32, block_size: int = 64, alloc_fac=1.0) -> jnp.ndarray:
    out_type = jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int32)

    nleaf = jnp.ones((posz.shape[0],), dtype=jnp.int32)
    xnleaf = jnp.concatenate((posz, nleaf[:,None].view(jnp.float32)), axis=-1)

    flag_split = jax.ffi.ffi_call("SummarizeLeaves", (out_type,), vmap_method="sequential")(
        xnleaf, len(posz), max_size=np.int32(leaf_size),
        block_size=np.uint64(block_size), scan_size=np.int32(leaf_size+1))[0]
    
    max_new_leaves = int(div_ceil(len(posz) * alloc_fac, np.maximum(leaf_size//2, 1)))
    
    splits = jnp.where(flag_split > -1000, size=max_new_leaves+1, fill_value=len(posz))[0]

    return splits
create_coarse_leaves.jit = jax.jit(create_coarse_leaves, static_argnames=("leaf_size", "block_size"))

def determine_znode_boundaries(posz: jnp.ndarray, block_size: int = 64, nleaves: jnp.array = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Builds a Z-order tree from positions"""
    if nleaves is None:
        nleaves = jnp.array(len(posz))

    out_types = (jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int32),)*3
    lvl, lbound, rbound = jax.ffi.ffi_call("FindNodeBoundaries", out_types)(
        posz, nleaves, block_size=np.uint64(block_size)
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

# ------------------------------------------------------------------------------------------------ #
#                                      Tree Building Functions                                     #
# ------------------------------------------------------------------------------------------------ #

def new_build_tree_hierarchy(part: PosMass, cfg: Config) -> NewTreeHierarchy:
    ispl =  create_coarse_leaves(part.pos, leaf_size=cfg.fmm.max_leaf_size, alloc_fac=cfg.fmm.alloc_fac_nodes)
    nleaves = jnp.argmax(ispl)
    lvl, lbound, rbound = determine_znode_boundaries(part.pos[ispl[:-1]], nleaves=nleaves)

    npart_node = ispl[rbound] - ispl[lbound]

    nlevels = np.log(len(ispl) / cfg.fmm.stop_coarsen) / np.log(cfg.fmm.coarse_fac)
    nlevels = np.maximum(int(np.ceil(nlevels)), 0)

    alloc_size = int(len(ispl) * cfg.fmm.alloc_fac_tree)
    
    ispl_n2l = PackedArray(jnp.zeros(alloc_size, dtype=jnp.int32), levels=nlevels)
    ispl_n2l = ispl_n2l.set(0, jnp.arange(alloc_size, dtype=jnp.int32), num=nleaves+1, fill_value=nleaves)
    ispl_n2n = PackedArray(jnp.zeros(alloc_size, dtype=jnp.int32), levels=nlevels)
    ispl_n2n = ispl_n2n.set(0, ispl, num=nleaves+1, fill_value=len(part.pos))

    def handle_level(i, carry):
        node_size = cfg.fmm.max_leaf_size * (cfg.fmm.coarse_fac ** i)

        ispl_n2l, ispl_n2n = carry
        
        # node to leaf relation
        new_nnodes = jnp.sum(npart_node > node_size)
        new_ispl_n2l = jnp.where(npart_node > node_size, size=ispl_n2n.size())[0]
        ispl_n2l = ispl_n2l.set(i, new_ispl_n2l, num=new_nnodes, fill_value=nleaves)

        # node to node relation
        last_n2l = ispl_n2l.get(i-1, ispl_n2n.size())
        last_npart = npart_node[last_n2l]
        new_ispl_n2n = jnp.where(last_npart > node_size, size=ispl_n2n.size())[0]
        ispl_n2n = ispl_n2n.set(i, new_ispl_n2n, num=new_nnodes, fill_value=ispl_n2n.num(i-1)-1)

        return ispl_n2l, ispl_n2n
    
    ispl_n2l, ispl_n2n = jax.lax.fori_loop(1, nlevels, handle_level, (ispl_n2l, ispl_n2n))

    # Check that the allocation was big enough
    def alloc_err(filled, size):
        raise RuntimeError(f"Tree allocation too small: filled {filled}, size {size}")
    
    ispl_n2l.ispl = ispl_n2l.ispl + conditional_callback(
        ispl_n2l.nfilled() > ispl_n2l.size(), alloc_err,
        ispl_n2l.nfilled(), ispl_n2l.size(),
    )

    ispl_n2p = ispl[ispl_n2l.data] # node to particle relation
    lvl, geom_cent, ext = get_node_geometry(part.pos, ispl_n2p[:-1], ispl_n2p[1:], num=ispl_n2l.nfilled()-1)

    # Can predict the shapes of further packed arrays
    leaf_array_spl = cumsum_starting_with_zero(ispl_n2l.ispl[1:]- ispl_n2l.ispl[:-1] - 1)

    if cfg.fmm.multipoles_around_com:
        def handle_mcent_level(i, carry):
            npos, nmass, posm = carry
            posm = center_of_mass(ispl_n2n.get(i, size=len(posm.mass)+1), posm)
            nmass = nmass.set(i, posm.mass, num=ispl_n2n.num(i)-1)
            npos = npos.set(i, posm.pos, num=ispl_n2n.num(i)-1)
            return npos, nmass, posm
        
        posm = center_of_mass(ispl, part)
        npos = PackedArray(posm.pos, ispl=leaf_array_spl, fill_values=jnp.nan)
        nmass = PackedArray(posm.mass, ispl=leaf_array_spl, fill_values=jnp.nan)

        mass_cent, nmass, _ = jax.lax.fori_loop(1, nlevels, handle_mcent_level, (npos, nmass, posm))
    else:
        mass_cent = None

    th = NewTreeHierarchy(
        ispl_n2n, ispl_n2l,
        lvl = PackedArray(lvl, leaf_array_spl, fill_values=-1000),
        geom_cent = PackedArray(geom_cent, leaf_array_spl, fill_values=jnp.nan),
        mass_cent = mass_cent
    )
    
    return th
new_build_tree_hierarchy.jit = jax.jit(new_build_tree_hierarchy, static_argnames=['cfg'])

def build_tree_hierarchy(part: PosMass, cfg: Config) -> Tuple[TreeHierarchy, list[TreePlane]]:
    ispl =  create_coarse_leaves(part.pos, leaf_size=cfg.fmm.max_leaf_size, alloc_fac=cfg.fmm.alloc_fac_nodes)
    nleaves = jnp.argmax(ispl)
    lvl, lbound, rbound = determine_znode_boundaries(part.pos[ispl[:-1]], nleaves=nleaves)

    mass_center = center_of_mass(ispl, part)

    th = TreeHierarchy(
        particles=part,
        leaf_ispl=ispl,
        leaf_mass_cent=mass_center,
        lbound=lbound,
        rbound=rbound,
        node_npart=ispl[rbound] - ispl[lbound]
    )
    
    tps = [th.leaf_plane()]
    last_node_size = cfg.fmm.max_leaf_size
    
    while tps[-1].size() > cfg.fmm.stop_coarsen:
        node_size = last_node_size * cfg.fmm.coarse_fac
        max_nodes = int(div_ceil(len(part.pos), np.maximum(node_size//2, 1)))
        tp = th.tree_plane(last_node_size, node_size, tps[-1].size(), max_nodes)
        tp.around_com = cfg.fmm.multipoles_around_com
        tps.append(tp)
        last_node_size = node_size
    
    return th, tps
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg'])