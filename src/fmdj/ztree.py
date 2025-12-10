import numpy as np
import jax
import jax.numpy as jnp

from fmdj_cuda import ffi_tree
from .tools import conditional_callback, div_ceil
from .data import TreePlane, PosMass
from .config import Config
from .multipoles import center_of_mass

jax.ffi.register_ffi_target("PosZorderSort", ffi_tree.PosZorderSort(), platform="CUDA")
jax.ffi.register_ffi_target("SummarizeLeaves", ffi_tree.SummarizeLeaves(), platform="CUDA")
jax.ffi.register_ffi_target("ZTreeNodeRelations", ffi_tree.ZTreeNodeRelations(), platform="CUDA")

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

def summarize_leaves(xleaf, nleaf=None, max_size=64, num_part=None, ref_fac=None, alloc_fac_nodes=1., alloc_min=128):
    """Summarizes leaf nodes into parent nodes
    """

    if nleaf is None:
        nleaf = jnp.ones((xleaf.shape[0],), dtype=jnp.int32)
        num = jnp.arange(0, len(xleaf)+1, dtype=jnp.int32)
    else:
        num = jnp.pad(jnp.cumsum(nleaf), (1, 0))
    if num_part is None:
        num_part = len(xleaf)
    if ref_fac is None:
        scan_size = max(max_size + 1, 16)
    else:
        assert len(xleaf) < num_part, "Don't specify ref_fac at lowest level"
        # Haven't perfectly understood yet, what scan_size is needed in the worst case...
        # In principle ref_fac + 2 should be enough, but I guess some rounding errors in the
        # ref_fac corrupt this sometimes (?) for now let's leave a bit slack
        scan_size = max(int(2.*ref_fac + 3), 16)
    assert scan_size <= 1024, "This is an absurd level of de-refinement. Please choose a smaller ref_fac or max_size"

    block_size = np.clip((scan_size//64) * 64, 64, 512)

    # We may have some invalid leaves at the end
    # Let's keep track until where our leaves are valid
    nleaves_filled = jnp.count_nonzero(nleaf)
    
    # This is an estimate of how many new leaves we may get. In general this is very robust and
    # may allocate up to 2x more leaves than needed.
    # However, for very imbalanced trees (may happen if the domain aligns very badly with the 
    # particle distribution) it may happen that we underestimate the number of new needed leaves.
    # This tends to be more likely when few nodes are left, so for most cases requiring a minimal
    # allocation size fixes it. However, to have a way out if things go wrong, we add an assertion
    # below and expose the allocation factors to the user.
    max_new_leaves = int(div_ceil(num_part, np.maximum(max_size//2, 1)))
    max_new_leaves = int(np.maximum(max_new_leaves, alloc_min) * alloc_fac_nodes)

    assert xleaf.dtype == jnp.float32
    assert xleaf.shape[-1] == 3

    xnleaf = jnp.concatenate((xleaf, nleaf[:,None].view(jnp.float32)), axis=-1)

    out_splits_type = jax.ShapeDtypeStruct((xnleaf.shape[0]+1,), jnp.int32)

    flag_split = jax.ffi.ffi_call("SummarizeLeaves", (out_splits_type,), vmap_method="sequential")(
        xnleaf, nleaves_filled, max_size=np.int32(max_size),
        block_size=np.uint64(block_size), scan_size=np.int32(scan_size))[0]
    
    # print(flag_split)
    # print(np.min(flag_split[flag_split>-1000]))

    # Get the splitting points of leaves
    splits = jnp.where(flag_split > -1000, size=max_new_leaves+1, fill_value=nleaves_filled)[0]
    new_nleaf = num[splits[1:]] - num[splits[:-1]]

    new_leaf_lvl = jnp.minimum(flag_split[splits[:-1]], flag_split[splits[1:]]) - 1

    numleaves = jnp.count_nonzero(new_nleaf)

    def assert_leaves_complete(nfilled, nshould):
        # If this check fails, it likely means that we did not predict a large enough allocation
        # for the new leaves. (See explanation in the comment above)
        assert nfilled == nshould, (f"Leaves not completely filled: {nfilled} != {nshould}."
                                    + "May be solved by increasing alloc_fac_nodes")
    conditional_callback(splits[-1] != nleaves_filled, assert_leaves_complete, splits[-1], nleaves_filled)

    new_leaf_cent = get_node_box(xleaf[splits[:-1]], jnp.full_like(new_leaf_lvl, new_leaf_lvl))[0]
    new_leaf_cent = jnp.where(jnp.arange(len(new_leaf_cent))[:,None] < numleaves, new_leaf_cent, jnp.nan)

    return splits, new_nleaf, new_leaf_lvl, new_leaf_cent, numleaves

summarize_leaves.jit = jax.jit(summarize_leaves, static_argnames=("max_size", "num_part", "ref_fac"))


from dataclasses import dataclass
@jax.tree_util.register_dataclass
@dataclass
class BinaryZTree:
    level: jnp.ndarray = None
    lbound: jnp.ndarray = None
    rbound: jnp.ndarray = None
    lchild: jnp.ndarray = None
    rchild: jnp.ndarray = None

def create_coarse_leaves(posz: jnp.ndarray, leaf_size: int = 32, block_size: int = 64) -> jnp.ndarray:
    out_type = jax.ShapeDtypeStruct((posz.shape[0]+1,), jnp.int32)

    nleaf = jnp.ones((posz.shape[0],), dtype=jnp.int32)
    xnleaf = jnp.concatenate((posz, nleaf[:,None].view(jnp.float32)), axis=-1)

    flag_split = jax.ffi.ffi_call("SummarizeLeaves", (out_type,), vmap_method="sequential")(
        xnleaf, len(posz), max_size=np.int32(leaf_size),
        block_size=np.uint64(block_size), scan_size=np.int32(leaf_size+1))[0]
    
    max_new_leaves = int(div_ceil(len(posz), np.maximum(leaf_size//2, 1)))
    
    splits = jnp.where(flag_split > -1000, size=max_new_leaves+1, fill_value=len(posz))[0]

    return splits
create_coarse_leaves.jit = jax.jit(create_coarse_leaves, static_argnames=("leaf_size", "block_size"))

def build_ztree(posz: jnp.ndarray, block_size: int = 64, leaf_size: int = 1, nleaves: int = None) -> BinaryZTree:
    """Builds a Z-order tree from positions"""
    if nleaves is None:
        nleaves = len(posz)

    out_type = jax.ShapeDtypeStruct((5, posz.shape[0]+1), jnp.int32)
    res = jax.ffi.ffi_call("ZTreeNodeRelations", (out_type,))(posz, block_size=np.uint64(block_size), nleaves=np.uint64(nleaves))[0]
    ztree = BinaryZTree(*res)

    root_node = jnp.argmax(ztree.level[1:-1]).astype(jnp.int32)
    ztree.level = ztree.level.at[0].set(root_node)
    ztree.level = ztree.level.at[-1].set(root_node)

    return ztree
build_ztree.jit = jax.jit(build_ztree)

# ------------------------------------------------------------------------------------------------ #
#                                      Tree Building Functions                                     #
# ------------------------------------------------------------------------------------------------ #

def coarsen_plane(fine: TreePlane, cfg : Config) -> TreePlane:
    """Gets the next coarser tree plane from a finer one"""
    max_size = int(fine.max_node_size * cfg.fmm.coarse_fac)
    
    res = summarize_leaves(
        fine.geom_cent, fine.npart, max_size=max_size, num_part=fine.tot_npart,
        ref_fac=cfg.fmm.coarse_fac, alloc_fac_nodes=cfg.fmm.alloc_fac_nodes
    )

    coarse = TreePlane(
        *res, 
        max_node_size = max_size, tot_npart = fine.tot_npart, 
        size_children = fine.size(), around_com=fine.around_com
    )
    coarse.mass_cent = center_of_mass(coarse.ispl, fine.mass_cent, cfg=cfg)


    return coarse
coarsen_plane.jit = jax.jit(coarsen_plane, static_argnames=['cfg'])

def build_tree_hierarchy(part: PosMass, cfg: Config) -> list[TreePlane]:
    res = summarize_leaves(
        part.pos, max_size=cfg.fmm.max_leaf_size, num_part=part.pos.shape[0],
        alloc_fac_nodes=cfg.fmm.alloc_fac_nodes
    )
    leaves = TreePlane(
        *res, 
        max_node_size=cfg.fmm.max_leaf_size, tot_npart=part.pos.shape[0], 
        size_children=len(part.pos), around_com=cfg.fmm.multipoles_around_com
    )
    leaves.mass_cent = center_of_mass(leaves.ispl, part, cfg=cfg)

    tree_levels : list[TreePlane] = [leaves]

    new_level = leaves
    while len(new_level.lvl) > cfg.fmm.stop_coarsen:
        new_level = coarsen_plane.jit(tree_levels[-1], cfg)
        tree_levels.append(new_level)
    return tree_levels
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg'])