import fmdj_ffi as cj
import jax
import jax.numpy as jnp
from fmdj.multipoles import x_moment, shift_multipoles, shift_local_to_local, ilist_node_to_node
from dataclasses import dataclass, field
from .config import Config, FMMConfig
from fmdj.tools import conditional_callback
from typing import Tuple, List
import fmdj
from .data import Multipoles, TreePlane, PosMass, SegmentedNDArray, InteractionList, dense_interaction_list
from .tools import offset_sum, masked_prefix_sum, inverse_of_splits, cumsum_starting_with_zero

from .variants import vm, make_dispatcher, V, VariantConfig

# from .multipoles import coarsen_multipoles
from fmdj_ffi.cj_new_tree import coarsen_multipoles, multipoles_from_particles

def coarsen_plane(fine: TreePlane, cfg : Config) -> TreePlane:
    """Gets the next coarser tree plane from a finer one"""
    cfg_tree: FMMConfig = cfg.fmm

    max_size = int(fine.max_node_size * cfg_tree.coarse_fac)
    
    res = cj.tree.summarize_leaves(
        fine.cent, fine.npart, max_size=max_size, num_part=fine.tot_npart,
        ref_fac=cfg_tree.coarse_fac, alloc_fac_nodes=cfg_tree.alloc_fac_nodes
    )

    coarse = TreePlane(*res, max_node_size = max_size, tot_npart = fine.tot_npart, size_children = fine.size())

    if fine.mp is not None:
        coarse.mp = coarsen_multipoles(fine.mp, coarse, cfg=cfg)

    return coarse
coarsen_plane.jit = jax.jit(coarsen_plane, static_argnames=['cfg'])

def build_tree_hierarchy(part: PosMass, cfg: Config) -> list[TreePlane]:
    cfg_tree: FMMConfig = cfg.fmm
    with_multipoles = cfg_tree.p > 0

    res = cj.tree.summarize_leaves(
        part.pos, max_size=cfg_tree.max_leaf_size, num_part=part.pos.shape[0],
        alloc_fac_nodes=cfg_tree.alloc_fac_nodes
    )
    leaves = TreePlane(*res, max_node_size=cfg_tree.max_leaf_size, tot_npart=part.pos.shape[0], size_children=len(part.pos))
    if with_multipoles:
        leaves.mp = multipoles_from_particles(leaves, part, cfg=cfg)

    tree_levels : list[TreePlane] = [leaves]

    new_level = leaves
    while len(new_level.lvl) > cfg_tree.stop_coarsen:
        new_level = coarsen_plane.jit(tree_levels[-1], cfg)
        tree_levels.append(new_level)
    return tree_levels
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                                         Interaction Lists                                        #
# ------------------------------------------------------------------------------------------------ #


def expand_interactions(
        ilist: InteractionList, 
        ispl: jnp.ndarray, 
        size_children: int, 
        size_new_ilist: int) -> InteractionList:
    """Expands the interaction list to the children"""
    # This works by adding two (variable size) extra dimensions to the interaction list.
    # Since the indexing logic of segmented arrays is rather complicated, we use a 
    # helper class SegmentedNDArray to handle the indexing.

    # Helper, Node->interaction
    seg_ilist = SegmentedNDArray(ispl=[ilist.ispl])

    # Node->Child segments
    seg_nodes = SegmentedNDArray(ispl=[ispl])

    # Expand to Node->Child->Nodeinteraction
    (in0, ic0), valid = seg_nodes.multi_indices(size_children, get_valid=True)
    node_exp = seg_nodes.expand(seg_ilist.n(in0) * valid)

    # Expand to Node->Child->Nodeinteraction->Otherchild
    # Note: this one is only needed temporarily and could in principle use a smaller size
    (in0, ic0, iint), valid = node_exp.multi_indices(size_new_ilist, get_valid=True)
    i1 = ilist.iother[seg_ilist.multi_to_flat((in0, iint))]
    node_cc = node_exp.expand(seg_nodes.n(i1) * valid)

    # Get interaction list
    (in0, ic0, iint, ic1), valid = node_cc.multi_indices(size_new_ilist, get_valid=True)
    inode1 = ilist.iother[seg_ilist.multi_to_flat((in0, iint))]
    iother_new = jnp.where(valid, seg_nodes.multi_to_flat((inode1, ic1)), size_new_ilist)

    # Discard last dimension
    ispln = node_cc.global_ispl(1)

    # Check that the sizes are big enough
    def size_error(nfilled, size):
        raise ValueError(f"Expanded interaction list ({nfilled}) does not fit into buffer ({size})")

    ispln = ispln + conditional_callback(ispln[-1] >= size_new_ilist, size_error, ispln[-1], size_new_ilist)
    
    return InteractionList(ispl = ispln, iother = iother_new, nfilled = ispln[-1])
expand_interactions.jit = jax.jit(expand_interactions, static_argnames=['size_children', 'size_new_ilist'])




# ------------------------------------------------------------------------------------------------ #
#                                             Tree Walk                                            #
# ------------------------------------------------------------------------------------------------ #

def norm2(dx: jnp.ndarray) -> jnp.ndarray:
    return dx[...,0]**2 + dx[...,1]**2 + dx[...,2]**2

def opening_criterion_bnh(plane: TreePlane, i0: jnp.ndarray, i1: jnp.ndarray, cfg: Config):
    """Barnes & Hut Opening Criterion."""
    theta = cfg.fmm.opening_angle

    r2 = norm2(plane.mp.center()[i1] - plane.mp.center()[i0])

    Lsum = plane.node_extent()[i0] + plane.node_extent()[i1]
    Lmax = jnp.maximum(jnp.maximum(Lsum[...,0], Lsum[...,1]), Lsum[...,2])

    need_open = Lmax*Lmax > theta**2 * r2
    # To avoid dealing with overflow issues, we open very large nodes explicitly:
    need_open = need_open | (plane.lvl[i1] >= 150) | (plane.lvl[i0] >= 150)

    return need_open

def _evaluate_plane_interactions_base(plane: TreePlane, 
                                plane_lr: TreePlane | None = None,
                                ilist_lr: InteractionList | None = None,
                                loc_lr: jnp.ndarray | None = None,
                                cfg: Config = None
                                ) -> Tuple[jnp.ndarray, InteractionList]:
    """Evaluate all interactions for a given tree plane."""
    if plane_lr is None or ilist_lr is None: # Root level
        ilist = dense_interaction_list(plane.size(), nnodes=plane.nnodes)
    else:
        # Add all child-child pairs for each coarser interaction
        ilist_size_new = cfg.fmm.ilist_alloc_fac * plane.size()
        ilist = expand_interactions(ilist_lr, plane_lr.ispl, plane.size(), ilist_size_new)

    # Interaction indices
    i0, i1, valid = ilist.get_interactions(get_valid=True)

    need_open = opening_criterion_bnh(plane, i0, i1, cfg=cfg)

    ilist_open = ilist.filter(valid & need_open)
    ilist_eval = ilist.filter(valid & ~need_open)
    
    # Evaluate multipole interactions
    interactions = jnp.stack(ilist_eval.get_interactions(get_valid=False), axis=-1)
    irange = jnp.stack([0, ilist_eval.nfilled])

    loc = ilist_node_to_node(plane.mp.center(), plane.mp.values, interactions, irange, cfg=cfg)
    if loc_lr is not None:
        ipar = plane_lr.icoarse_of_fine()
        loc = loc + shift_local_to_local(loc_lr[ipar], plane.mp.center() - plane_lr.mp.center()[ipar])

    # Some logging
    open_frac = ilist_open.nfilled / ilist.nfilled
    fmdj.log("Interactions opened {}/{} ({:.1%}) sizefac: {:.1f} ({:.1%} of allocation)", 
             ilist_open.nfilled, ilist.nfilled, open_frac, ilist.nfilled / plane.size(), 
             ilist.nfilled / ilist.size(), level=2, cfg=cfg)
    
    return loc, ilist_open
_evaluate_plane_interactions_base.jit = jax.jit(_evaluate_plane_interactions_base, static_argnames=['cfg'])

def evaluate_interaction_hierarchy(th, cfg):
    ilist, loc, last_plane = None, None, None
    for i in reversed(range(0, len(th))):
        loc, ilist = evaluate_plane_interactions(th[i], last_plane, ilist, loc, cfg=cfg)
        last_plane = th[i]
    return loc, ilist
evaluate_interaction_hierarchy.jit = jax.jit(evaluate_interaction_hierarchy, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                                       Register Dispatchers                                       #
# ------------------------------------------------------------------------------------------------ #

evaluate_plane_interactions = make_dispatcher(vm[V.evaluate_plane_interactions], _evaluate_plane_interactions_base, add_jit=True)

# ------------------------------------------------------------------------------------------------ #
#                                             New walk                                             #
# ------------------------------------------------------------------------------------------------ #

def invert_splits(ispl, size, weights=1):
    iarr = jnp.arange(size)
    mask = jnp.zeros_like(iarr).at[ispl[1:]].add(weights)
    return jnp.cumsum(mask)

def get_double_index(ispl, size, absolute=False):
    i0 = invert_splits(ispl, size)
    if absolute:
        return i0, jnp.arange(size, dtype=ispl.dtype)
    else:
        ioff = invert_splits(ispl, size, weights=(ispl[1:] - ispl[:-1]))
        return i0, jnp.arange(size, dtype=ispl.dtype) - ioff


def div_ceil(a, b):
    return (a + b - 1) // b

def fori_dynamic_over_static(lower, upper, body_fun, init_val, *, unroll=None, nstatic=None):
    """Does a loop with a dynamical boundary over a loop with static boundaries

    This function can have two advantages over a standard jax.lax.fori_loop with dynamic
    boundaries: (1) it allows you to unroll the (inner) static loop partially. (E.g. try unroll=4)
    (2) Static loops seem to be a lot faster in jax. Honestly, I don't know why!

    The usage is the same as jax.fori_loop, with the important difference that the loop
    may also be executed a few extra times if (upwer-lower) is not divisible by nstatic.
    So be sure to discard invalid iterations in your function!

    Example:
    def f(iter, x):
        return jnp.where(iter < 10, x + 1, x)

    print(fori_dynamic_over_static(0, 10, f, 0., nstatic=7))
    """

    if nstatic is None:
        return jax.lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)
    
    
    def outer_body(iouter, state):
        def inner_body(iinner, state):
            return body_fun(lower + iouter*nstatic + iinner, state)
        return jax.lax.fori_loop(0, nstatic, inner_body, state, unroll=unroll)

    ndynamic = div_ceil(upper - lower, nstatic)
    return jax.lax.fori_loop(0, ndynamic, outer_body, init_val)


def new_eval(
        plane: TreePlane, 
        plane_lr: TreePlane | None = None,
        ilist_lr: InteractionList | None = None,
        loc_lr: jnp.ndarray | None = None,
        cfg: Config = None
    ) -> Tuple[jnp.ndarray, InteractionList]:
    unroll = cfg.old.interact_unroll

    spl_nodes = plane_lr.ispl
    spl_int = ilist_lr.ispl
    node_size = spl_nodes[1:] - spl_nodes[:-1]

    int_p1 = ilist_lr.iother
    ilist_size = plane.size() * cfg.fmm.ilist_alloc_fac

    # Pre-calculate some variables
    i0 = jnp.arange(plane.size(), dtype=jnp.int32)
    iparent = invert_splits(spl_nodes, plane.size())
    irange = jnp.stack((jnp.array(0), plane.nnodes))

    # Calculate loop boundaries
    ip0, ip1, valid = ilist_lr.get_interactions(get_valid = True)
    nint = jax.ops.segment_sum(node_size[ip1]*valid, ip0, num_segments=plane.size())

    def handle_counters(iint, isub):
        ip1 = int_p1[iint]
        i1 = spl_nodes[ip1] + isub
        valid = iint < spl_int[iparent+1]

        next_parent_interaction = isub+1 >= node_size[ip1]
        iint = jnp.where(next_parent_interaction, iint+1, iint)
        isub = jnp.where(next_parent_interaction, 0, isub+1)

        return i1, valid, iint, isub

    def loop_count(iter, state):
        nopen, Loc, iint, isub = state[0:4]

        i1, valid, iint, isub = handle_counters(iint, isub)

        # Count nodes that need to be opened
        need_open = opening_criterion_bnh(plane, i0, i1, cfg=cfg)
        nopen += need_open & valid

        # Evaluate M2L for unopened nodes
        interactions = jnp.stack((i0, i1), axis=-1)
        Loc_new = ilist_node_to_node(plane.mp.center(), plane.mp.values, interactions, irange, cfg=cfg)
        Loc = Loc + Loc_new * (valid & ~need_open)[:,None]

        return nopen, Loc, iint, isub

    nopen = jnp.zeros(plane.size(), dtype=jnp.int32)
    Loc = jnp.zeros_like(plane.mp.values)

    nopen, Loc = fori_dynamic_over_static(
        0, jnp.max(nint),
        loop_count,
        (nopen, Loc, spl_int[iparent], jnp.zeros_like(iparent)),
        unroll = unroll,
        nstatic = 64
    )[0:2]
    offsets = cumsum_starting_with_zero(nopen)

    def loop_insert(iter, state):
        nopen, new_ilist, iint, isub = state[0:4]

        i1, valid, iint, isub = handle_counters(iint, isub)

        # Count nodes that need to be opened
        need_open = opening_criterion_bnh(plane, i0, i1, cfg=cfg)

        # Insert interactions 
        iupdate = jnp.where(need_open & valid, offsets[:-1] + nopen, new_ilist.size)
        new_ilist = new_ilist.at[iupdate].set(i1)

        nopen += need_open & valid

        return nopen, new_ilist, iint, isub

    new_ilist = jnp.zeros(ilist_size, dtype=jnp.int32)
    nopen, new_ilist = fori_dynamic_over_static(
        0, jnp.max(nint),
        loop_insert,
        (jnp.zeros_like(nopen), new_ilist, spl_int[iparent], jnp.zeros_like(iparent)),
        unroll = unroll,
        nstatic = 128
    )[0:2]

    new_ilist = InteractionList(offsets, iother = new_ilist, nfilled = offsets[-1])

    if loc_lr is not None:
        Loc = Loc + shift_local_to_local(loc_lr[iparent], plane.mp.center() - plane_lr.mp.center()[iparent])

    # Some logging
    nfilled, ntot = offsets[-1], jnp.sum(nint*node_size[iparent])
    fmdj.log("Interactions opened {}/{} ({:.1%}) sizefac: {:.1f} ({:.1%} of allocation)", 
             nfilled, ntot, nfilled/ntot, new_ilist.nfilled/plane.size(), 
             nfilled/new_ilist.size(), level=2, cfg=cfg)

    return Loc, new_ilist
new_eval.jit = jax.jit(new_eval, static_argnames="cfg")