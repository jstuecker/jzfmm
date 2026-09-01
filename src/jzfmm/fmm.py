from typing import Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import warnings
from dataclasses import dataclass, replace


from jztree.data import PosMass, InteractionList, get_pos_mass, TreeHierarchy, PosLvl, PackedArray, RankIdx, get_num
from jztree.tree import grouped_dense_interaction_list, zsort_and_tree
from jztree.tree import distr_grouped_dense_interaction_list, simplify_interaction_list
from jztree.jax_ext import empty_like, pcast_like, raise_if, shard_map_constructor
from jztree.tools import masked_inverse
from jztree.comm import all_to_all_request_children, all_to_all_with_irank, get_rank_info, in_shard_map_context
from jztree.stats import AllocStats, stats_callback
from jax.sharding import PartitionSpec as P

from .config import DirectSummationConfig, FMMConfig
from .data import LocalExpansion
from .multipoles import _fmm_node_to_child, build_multipole_hierarchy, num_multi, p_of_num_multi, shift_local_to_children_vjp_x

import jzfmm_cuda.ffi_fmm as ffi_fmm
import jzfmm_cuda.ffi_pair_summation as ffi_pair_summation
jax.ffi.register_ffi_target("CountInteractionsAndM2L", ffi_fmm.CountInteractionsAndM2L(), platform="CUDA")
jax.ffi.register_ffi_target("InsertInteractions", ffi_fmm.InsertInteractions(), platform="CUDA")
jax.ffi.register_ffi_target("LeafLeafPairSummation", ffi_pair_summation.LeafLeafPairSummation(), platform="CUDA")
jax.ffi.register_ffi_target("BwdLeafLeafPairSummation", ffi_pair_summation.BwdLeafLeafPairSummation(), platform="CUDA")
jax.ffi.register_ffi_target("DirectPairSummation", ffi_pair_summation.DirectPairSummation(), platform="CUDA")
jax.ffi.register_ffi_target("BwdDirectPairSummation", ffi_pair_summation.BwdDirectPairSummation(), platform="CUDA")

# ------------------------------------------------------------------------------------------------ #
#                                          M2L Evaluation                                          #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FMMChildData:
    poslvl: PosLvl
    mp: jax.Array

def _comm_buffer_size(base_size: int, factor: float) -> int:
    return max(base_size, int(np.ceil(base_size * factor)))

def _fmm_node_to_node(
        node_range: jax.Array,
        node_ilist: InteractionList,
        spl_recv: jax.Array,
        child_recv: FMMChildData,
        cfg_fmm: FMMConfig,
        single_thread_per_receiver: jax.Array | None = None,
        loc_in: jax.Array | None = None,
        spl_src: jax.Array | None = None,
        child_src: FMMChildData | None = None
    ) -> Tuple[jax.Array, InteractionList]:
    """
    Evaluates M2L for a tree plane and generates child interaction list.

    single_thread_per_receiver: Use this to make results exactly independent of grouping
       of nodes. Only useful to use at top-level to achieve GPU count independence.

    Outputs:
    - loc: multipole local expansion, shape (Nchild, M)
    - new_ilist: new interaction list for children
    """
    if spl_src is None:
        spl_src = spl_recv
    if child_src is None:
        child_src = child_recv
    if single_thread_per_receiver is None:
        single_thread_per_receiver = jnp.zeros(1, dtype=jnp.int32)
    else:
        single_thread_per_receiver = jnp.asarray(single_thread_per_receiver, dtype=jnp.int32)
    node_range = jnp.asarray(node_range, dtype=jnp.int32)
    spl_recv = jnp.asarray(spl_recv, dtype=jnp.int32)
    spl_src = jnp.asarray(spl_src, dtype=jnp.int32)
    ilist_ispl = jnp.asarray(node_ilist.ispl, dtype=jnp.int32)
    ilist_isrc = jnp.asarray(node_ilist.isrc, dtype=jnp.int32)

    size = len(child_recv.poslvl.pos)
    ilist_alloc_size = node_ilist.size()
    kernel = cfg_fmm.kernel
    dim = child_recv.poslvl.pos.shape[-1]

    if child_src.mp.dtype == jnp.float64 and cfg_fmm.p >= 6:
        warnings.warn(
            "For double precision, M2L order 6 and above currently spills registers "
            "and can be substantially slower than order 5.",
            RuntimeWarning,
            stacklevel=2,
        )

    assert ilist_alloc_size < 2**31, (
        f"So far only int32 is supported ({ilist_alloc_size / 2**31:.3g} * 2**31 entries requested)"
    )
    assert len(spl_recv) == len(node_ilist.ispl)

    # children = jnp.concatenate((plane.center(), plane.lvl.view(jnp.float32)[...,None]), axis=-1)
    children_recv = child_recv.poslvl.pos_lvl()
    children_src = child_src.poslvl.pos_lvl()
    
    # Determine output shapes
    dtype = child_src.mp.dtype
    kernel_params = kernel.params(dtype=dtype)
    opening = cfg_fmm.opening
    opening_params = opening.params(dtype=dtype)
    out_loc = jax.ShapeDtypeStruct((size, num_multi(cfg_fmm.p, dim=dim)), dtype)
    if loc_in is None:
        loc_in = jnp.zeros(out_loc.shape, dtype=out_loc.dtype)
        loc_in = pcast_like(loc_in, spl_recv)
    out_interaction_count = jax.ShapeDtypeStruct((size,), jnp.int32)
    
    # Count opened interactions and evaluate M2L
    loc, interaction_counts = jax.ffi.ffi_call(
        "CountInteractionsAndM2L",
        (out_loc, out_interaction_count,),
        input_output_aliases={11: 0},
    )(
        node_range, spl_recv, spl_src, ilist_ispl, ilist_isrc,
        children_recv, children_src, child_src.mp, kernel_params, opening_params,
        single_thread_per_receiver, loc_in,
        p=np.int32(cfg_fmm.p),
        radial_kernel_kind=np.int32(kernel.kind_id()),
        opening_criterion_kind=np.int32(opening.kind_id()),
    )
    loc, interaction_counts = jax.tree.map(
        lambda x: pcast_like(x, spl_recv), (loc, interaction_counts)
    )
    # valid_child = jnp.arange(size, dtype=spl_recv.dtype) < spl_recv[-1]
    # interaction_counts = jnp.where(valid_child, interaction_counts, 0)

    # Insert interactions
    ispl_child = jnp.pad(jnp.cumsum(interaction_counts), (1, 0)).astype(jnp.int32)
    out_child_ilist = jax.ShapeDtypeStruct((ilist_alloc_size,), jnp.int32)

    child_ilist = jax.ffi.ffi_call(
        "InsertInteractions",
        (out_child_ilist,)
    )(
        node_range, spl_recv, spl_src, ilist_ispl, ilist_isrc,
        children_recv, children_src, ispl_child, opening_params,
        opening_criterion_kind=np.int32(opening.kind_id()),
    )[0]
    child_ilist = pcast_like(child_ilist, spl_recv)

    # Create interaction list from outputs
    new_ilist = InteractionList(
        ispl=ispl_child,
        isrc=child_ilist,
        has_separate_query_and_source_indices=(
            node_ilist.has_separate_query_and_source_indices
        ),
    )

    new_ilist.ispl = new_ilist.ispl + raise_if(
        new_ilist.nfilled() > new_ilist.size(), 
        "Interaction list allocation too small ({nfil}/{size})\nHint: Increase alloc_fac_ilist", 
        nfil=new_ilist.nfilled(), size=new_ilist.size()
    )
    stats_callback(
        "allocation", AllocStats.record_filled_interactions,
        new_ilist.nfilled(), new_ilist.size()
    )
    
    return loc, new_ilist
_fmm_node_to_node.jit = jax.jit(_fmm_node_to_node, static_argnames=['cfg_fmm'])

def _fmm_dual_walk(th: TreeHierarchy, mph: PackedArray, cfg_fmm: FMMConfig):
    in_smap = in_shard_map_context()
    if in_smap:
        rank, ndev, axis_name = get_rank_info()

    # define root level:
    size = th.size()
    comm_size_nodes = _comm_buffer_size(size, cfg_fmm.alloc_fac_comm_nodes)
    center0 = th.center().get(0, size)
    dim = center0.shape[-1]
    if in_smap:
        spl, ilist, nsup = distr_grouped_dense_interaction_list(
            th.num(th.num_planes()-1), size,
            size_ilist=int(th.size_leaves*cfg_fmm.alloc_fac_ilist),
            size_ids=comm_size_nodes,
            separate_query_and_source_indices=True,
        )
    else:
        spl, ilist, nsup = grouped_dense_interaction_list(
            th.num(th.num_planes()-1), size_ilist=int(th.size_leaves*cfg_fmm.alloc_fac_ilist),
            ngroup=32, size_super=size
        )

    loc = jnp.zeros((size, num_multi(cfg_fmm.p, dim=dim)), dtype=mph.data.dtype)
    if in_smap:
        loc = pcast_like(loc, mph.data)

    # Add super node data into tree information
    spl_n2n = th.ispl_n2n.append(spl, nsup+1, fill_value=spl[-1], resize=True)
    cent = th.center().append(jnp.zeros((size, dim), dtype=center0.dtype), nsup, fill_value=0., resize=True)
    top_level = th.num_planes() - 1
    super_level = jnp.max(th.lvl.get(
        top_level, size, fill_value=jnp.iinfo(jnp.int32).min
    ))
    lvl = th.lvl.append(
        jnp.full(size, super_level, dtype=jnp.int32), nsup,
        fill_value=super_level, resize=True,
    )

    def handle_level(i, carry):
        level = th.num_planes() - 1 - i

        parent_loc, parent_ilist = carry
        parent_spl_recv = spl_n2n.get(level+1, size+1)
        parent_cent = cent.get(level+1, size)
        single_thread_per_receiver = jnp.asarray(i == 0, dtype=jnp.int32)[None]

        child_recv = FMMChildData(
            poslvl=th.poslvl(level, size),
            mp=mph.get(level, size=size)
        )

        if in_smap:
            # Request the remote source node data that local receivers interact with.
            # This also runs in size-one shard maps as a syntax/shape test path.
            (child_src, ids), parent_spl_src, dev_spl = all_to_all_request_children(
                parent_ilist.dev_spl, parent_ilist.ids, parent_spl_recv,
                (child_recv, jnp.arange(size, dtype=jnp.int32)),
                output=empty_like((child_recv, jnp.arange(size, dtype=jnp.int32)), new_size=comm_size_nodes),
                axis_name=axis_name, err_hint_child="\nHint: increase alloc_fac_comm_nodes",
                err_hint_parent="\nHint: increase alloc_fac_comm_nodes"
            )
            stats_callback(
                "allocation", AllocStats.record_filled_nodes_interaction,
                dev_spl[-1], size
            )
            parent_range = jnp.array([0, parent_ilist.ispl.size-1], dtype=jnp.int32)
        else:
            child_src, parent_spl_src = child_recv, parent_spl_recv
            parent_range = jnp.array([0, spl_n2n.num(level+1)-1], dtype=jnp.int32)

        loc_ch = _fmm_node_to_child(
            parent_spl_recv, parent_loc,
            PosLvl(pos=parent_cent, lvl=lvl.get(level+1, size)),
            child_recv.poslvl, cfg_fmm=cfg_fmm
        )

        loc_ch, ilist = _fmm_node_to_node(
            parent_range, parent_ilist, parent_spl_recv, child_recv,
            cfg_fmm=cfg_fmm, single_thread_per_receiver=single_thread_per_receiver,
            loc_in=loc_ch,
            spl_src=parent_spl_src, child_src=child_src
        )

        if in_smap:
            ilist = replace(ilist, ids=ids, dev_spl=dev_spl)
            ilist = simplify_interaction_list(ilist)

        return loc_ch, ilist

    loc, ilist = jax.lax.fori_loop(
        0, th.num_planes(), handle_level, (loc, ilist), unroll=True
    )

    return loc, ilist
_fmm_dual_walk.jit = jax.jit(_fmm_dual_walk, static_argnames=['cfg_fmm'])

# ------------------------------------------------------------------------------------------------ #
#                           Leaf-Leaf (Particle to Particle) Interactions                          #
# ------------------------------------------------------------------------------------------------ #

def _sum_to_input_shape(x: jax.Array, target: jax.Array) -> jax.Array:
    if x.shape == jnp.shape(target):
        return x
    return jnp.reshape(jnp.sum(x), jnp.shape(target))

def leaf_leaf_summation(
        particles: PosMass,
        ispl: jax.Array,
        ilist: InteractionList,
        cfg_fmm: FMMConfig = None,
        loc_in: jax.Array | None = None,
    ) -> jax.Array:
    particles_recv = particles
    spl_recv = jnp.asarray(ispl, dtype=jnp.int32)
    comm_size_particles = _comm_buffer_size(particles_recv.pos.shape[0], cfg_fmm.alloc_fac_comm_particles)
    in_smap = in_shard_map_context()
    if in_smap:
        rank, _, axis_name = get_rank_info()

    block_size = 128
    assert cfg_fmm.tree.max_leaf_size <= block_size
    dim = particles_recv.pos.shape[-1]

    node_range = jnp.array([0, spl_recv.size-1], dtype=jnp.int32)
    posm_recv = get_pos_mass(particles_recv)
    out_type = jax.ShapeDtypeStruct((particles_recv.pos.shape[0], dim + 1), posm_recv.dtype)
    if loc_in is None:
        loc_in = jnp.zeros(out_type.shape, dtype=out_type.dtype)
        loc_in = pcast_like(loc_in, spl_recv)
    kernel = cfg_fmm.kernel
    kernel_params = kernel.params(dtype=posm_recv.dtype)

    def eval_fwd(particles_recv, spl_recv, ilist, loc_in):
        spl_recv = jnp.asarray(spl_recv, dtype=jnp.int32)
        ilist = replace(
            ilist,
            ispl=jnp.asarray(ilist.ispl, dtype=jnp.int32),
            isrc=jnp.asarray(ilist.isrc, dtype=jnp.int32),
        )
        if in_smap:
            particles_src, spl_src, _dev_spl_src = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, spl_recv, particles_recv,
                output=empty_like(particles_recv, new_size=comm_size_particles),
                axis_name=axis_name,
                err_hint_parent="\nHint: increase alloc_fac_comm_nodes.",
                err_hint_child="\nHint: increase alloc_fac_comm_particles."
            )
            stats_callback(
                "allocation", AllocStats.record_filled_part_interactions,
                _dev_spl_src[-1], particles_src.pos.shape[0]
            )
        else:
            particles_src, spl_src = particles_recv, spl_recv
        spl_src = jnp.asarray(spl_src, dtype=jnp.int32)

        loc = jax.ffi.ffi_call(
            "LeafLeafPairSummation", (out_type,),
            input_output_aliases={8: 0},
        )(
            node_range, spl_recv, spl_src, ilist.ispl, ilist.isrc,
            get_pos_mass(particles_recv), get_pos_mass(particles_src), kernel_params, loc_in,
            radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
            kahan=bool(cfg_fmm.kahan_summation),
            remove_self_interaction=bool(cfg_fmm.remove_self_interaction),
        )[0]
        loc = pcast_like(loc, spl_recv)
        num = getattr(particles_recv, "num", None)
        if num is not None:
            valid = jnp.arange(loc.shape[0], dtype=jnp.int32) < num
            loc = jnp.where(valid[:, None], loc, jnp.nan)
        return loc, (particles_recv, spl_recv, ilist, particles_src, spl_src)

    @jax.custom_vjp
    def eval(particles_recv, spl_recv, ilist, loc_in):
        return eval_fwd(particles_recv, spl_recv, ilist, loc_in)[0]
    
    def eval_bwd(res, gloc):
        particles_recv, spl_recv, ilist, particles_src, spl_src = res
        spl_recv = jnp.asarray(spl_recv, dtype=jnp.int32)
        spl_src = jnp.asarray(spl_src, dtype=jnp.int32)
        ilist = replace(
            ilist,
            ispl=jnp.asarray(ilist.ispl, dtype=jnp.int32),
            isrc=jnp.asarray(ilist.isrc, dtype=jnp.int32),
        )
        if in_smap:
            gloc_src, _, _ = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, spl_recv, gloc,
                output=empty_like(gloc, float_val=0., new_size=comm_size_particles),
                axis_name=axis_name,
                err_hint_parent="\nHint: increase alloc_fac_comm_nodes",
                err_hint_child="\nHint: increase alloc_fac_comm_particles."
            )
        else:
            gloc_src = gloc

        # The globally symmetric leaf interaction list lets each owner compute its particle
        # gradients in one pass by requesting source particles and source cotangents.
        gposm = jax.ffi.ffi_call("BwdLeafLeafPairSummation", (out_type,))(
            node_range, spl_recv, spl_src, ilist.ispl, ilist.isrc,
            get_pos_mass(particles_recv), get_pos_mass(particles_src),
            kernel_params, gloc, gloc_src,
            radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
            kahan=bool(cfg_fmm.kahan_summation),
            remove_self_interaction=bool(cfg_fmm.remove_self_interaction),
        )[0]
        gpos = pcast_like(gposm[:,:dim], particles_recv.pos)
        gmass = pcast_like(
            _sum_to_input_shape(gposm[:,dim], particles_recv.mass),
            particles_recv.mass,
        )
        gnum = None
        if particles_recv.num is not None:
            gnum = jnp.zeros_like(particles_recv.num, dtype=jax.dtypes.float0)

        gposm = PosMass(
            pos=gpos, mass=gmass, num=gnum, num_total=particles_recv.num_total
        )
        return gposm, None, None, gloc
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(particles_recv, spl_recv, ilist, loc_in)
leaf_leaf_summation.jit = jax.jit(leaf_leaf_summation, static_argnames=['cfg_fmm'])


# ------------------------------------------------------------------------------------------------ #
#                                      Direct Summation Forces                                     #
# ------------------------------------------------------------------------------------------------ #

def direct_summation(
        part: PosMass,
        cfg_direct: DirectSummationConfig,
        G=1,
    ) -> LocalExpansion:
    block_size = 64
    posm = get_pos_mass(part)
    out_type = jax.ShapeDtypeStruct(posm.shape, posm.dtype)
    kernel = cfg_direct.kernel
    kernel_params = kernel.params(dtype=posm.dtype)
    
    @jax.custom_vjp
    def eval(xm):
        loc = jax.ffi.ffi_call("DirectPairSummation", (out_type,))(
        xm, kernel_params, block_size=np.uint64(block_size),
        radial_kernel_kind=np.int32(kernel.kind_id()), kahan=cfg_direct.kahan_summation,
        remove_self_interaction=bool(cfg_direct.remove_self_interaction))[0]
        return loc
    def eval_fwd(xm):
        return eval(xm), xm
    def eval_bwd(xm, gloc):
        gxm = jax.ffi.ffi_call("BwdDirectPairSummation", (out_type,))(
            gloc, xm, kernel_params, block_size=np.uint64(block_size),
            radial_kernel_kind=np.int32(kernel.kind_id()), kahan=cfg_direct.kahan_summation,
            remove_self_interaction=bool(cfg_direct.remove_self_interaction)
        )[0]
        return gxm,
    
    eval.defvjp(eval_fwd, eval_bwd)

    return LocalExpansion(eval(posm) * G, dim=part.pos.shape[-1])
direct_summation.jit = jax.jit(direct_summation, static_argnames=['cfg_direct'])

def direct_potential_jax(x, m=1., softening=1e-2):
    rij2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(rinv * jnp.broadcast_to(m, x.shape[:-1])[None,:], axis=1)
direct_potential_jax.jit = jax.jit(direct_potential_jax)

def direct_force_jax(x, m=1., softening=1e-2):
    dx = x[:, None] - x[None, :]
    rij2 = jnp.sum(dx ** 2, axis=-1, keepdims=True)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(dx * rinv**3 * jnp.broadcast_to(m, x.shape[:-1])[None,:,None], axis=1)
direct_force_jax.jit = jax.jit(direct_force_jax)

def direct_potential_scan_jax(x, m=1., n2lim=1e8, eps=1e-5):
    N = x.shape[0]

    nmax = int(np.ceil(n2lim / len(x)))
    nev = int(np.ceil(x.shape[0] / nmax))

    def potential_over_range(i1, i2):
        xi = x[jnp.arange(nmax, dtype=jnp.int32) + i1]
        # Compute vector distances to all other particles
        r_ij2 = jnp.sum((x - xi[:,None])**2, axis=-1)
        distinv = jnp.where(r_ij2 < 1e-30, 0., 1./jnp.sqrt(r_ij2 + eps**2)) # avoid self-interaction

        return - jnp.sum(m * distinv, axis=1)

    def handle_interval(_, i):
        return None, potential_over_range(nmax * i, nmax * (i + 1))

    _, phis = jax.lax.scan(handle_interval, None, jnp.arange(nev, dtype=jnp.int32))

    return jnp.concatenate(phis)[0:N]
direct_potential_scan_jax.jit = jax.jit(direct_potential_scan_jax, static_argnames=("n2lim",))

# ------------------------------------------------------------------------------------------------ #
#                                         Master Functions                                         #
# ------------------------------------------------------------------------------------------------ #


def evaluate_node_node_fmm(partz: PosMass, th: TreeHierarchy, *, cfg_fmm: FMMConfig) -> Tuple[jax.Array, InteractionList]:
    
    def eval_fwd(pos, mp, pout=1):
        mph = build_multipole_hierarchy(th, pos, mp, cfg_fmm=cfg_fmm)
        loc_node, ilist = _fmm_dual_walk(th, mph, cfg_fmm=cfg_fmm)
        ispl = th.splits_leaf_to_part()
        node = th.poslvl(0)
        # Particle scale H=1 makes the L2P output use physical derivative units.
        particle = PosLvl(pos=pos, lvl=jnp.zeros(pos.shape[0], dtype=jnp.int32))
        loc_part = _fmm_node_to_child(ispl, loc_node, node, particle, pout=pout, cfg_fmm=cfg_fmm)
        return (loc_part, ilist), (pos, mp, ispl, node, loc_node)
    
    def eval_bwd(pout, res, grads):
        pos, mp, ispl, node, loc_node = res
        gloc = grads[0]
        particle = PosLvl(pos=pos, lvl=jnp.zeros(pos.shape[0], dtype=jnp.int32))

        # Backwards pass = FMM with gloc as multipole weights
        # plus the position derivatives of the shifting operators

        gx1 = shift_local_to_children_vjp_x(ispl, loc_node, node, particle, gloc)
        
        dim = pos.shape[-1]
        (gmp, _), (_, _, _, _, gmp_node) = eval_fwd(pos, gloc, pout=p_of_num_multi(mp.shape[-1], dim=dim))

        gx2 = shift_local_to_children_vjp_x(ispl, gmp_node, node, particle, mp)
        
        return gx1 + gx2, _sum_to_input_shape(gmp, mp)
    
    @partial(jax.custom_vjp, nondiff_argnames=['pout'])
    def eval(pos, mp, pout=1):
        return eval_fwd(pos, mp, pout=pout)[0]
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(partz.pos, jnp.reshape(partz.mass, jnp.shape(partz.mass) + (1,)))
evaluate_node_node_fmm.jit = jax.jit(evaluate_node_node_fmm, static_argnames=['cfg_fmm', ])

def _fast_multipole_method_z(
        partz: PosMass, th: TreeHierarchy, *, cfg_fmm: FMMConfig, G=1., pout: int = 1
    ) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."

    loc_node_node, ilist = evaluate_node_node_fmm(partz, th, cfg_fmm=cfg_fmm)
    spl = th.splits_leaf_to_part()

    loc = leaf_leaf_summation(partz, spl, jax.lax.stop_gradient(ilist), cfg_fmm=cfg_fmm, loc_in=loc_node_node)

    return LocalExpansion(loc * G, dim=partz.pos.shape[-1])
_fast_multipole_method_z.jit = jax.jit(_fast_multipole_method_z, static_argnames=("cfg_fmm", "G", "pout"))

def _parse_fmm_result(result: str) -> Tuple[str, ...]:
    keys = tuple(result.split("_"))
    valid = {"loc", "locz", "partz", "tree"}
    unknown = set(keys) - valid
    if unknown:
        raise ValueError(f"Unknown FMM result key(s): {sorted(unknown)}. Valid keys are {sorted(valid)}.")
    return keys

def _as_posmass(part) -> PosMass:
    return PosMass(
        pos=part.pos, mass=part.mass,
        num=getattr(part, "num", None),
        num_total=getattr(part, "num_total", None)
    )

def fast_multipole_method(
        part: PosMass, cfg_fmm: FMMConfig, th: TreeHierarchy | None = None,
        result: str = "loc", G=1., pout: int = 1
    ) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."
    keys = _parse_fmm_result(result)
    in_smap = in_shard_map_context()

    if th is None:
        size = part.pos.shape[0]
        idx = jnp.arange(size, dtype=jnp.int32)
        if in_smap:
            rank, ndev, axis_name = get_rank_info()
            origin = RankIdx(rank=jnp.full(size, rank, dtype=jnp.int32), idx=idx)
        else:
            ndev = 1
            origin = RankIdx(rank=None, idx=idx)
        num_origin = get_num(part, default_to_length=(ndev == 1))
        partz, dataz, th = zsort_and_tree(
            part, cfg_fmm.tree, data=origin
        )
        origin_z = dataz
    elif "loc" in keys:
        raise ValueError(
            "result='loc' is only available when fast_multipole_method builds the tree. "
            "Use result='locz' when providing a tree."
        )
    else:
        partz = part
        origin_z = None
        num_origin = None

    locz = None
    if ("loc" in keys) or ("locz" in keys):
        locz = _fast_multipole_method_z(_as_posmass(partz), th, cfg_fmm=cfg_fmm, G=G, pout=pout)

    loc = None
    if "loc" in keys:
        if in_smap:
            (loc_values, idx), _dev_spl = all_to_all_with_irank(
                origin_z.rank, (locz.values, origin_z.idx), num=partz.num,
                axis_name=axis_name, err_hint="\nThis should never fail..."
            )
        else:
            loc_values, idx = locz.values, origin_z.idx
        inv_sort = masked_inverse(idx, mask=jnp.arange(len(idx), dtype=idx.dtype) < num_origin)
        loc = LocalExpansion(loc_values[inv_sort], dim=part.pos.shape[-1])

    out = {
        "loc": loc,
        "locz": locz,
        "partz": partz,
        "tree": th,
    }
    res = tuple(out[key] for key in keys)
    return res[0] if len(res) == 1 else res
fast_multipole_method.jit = jax.jit(fast_multipole_method, static_argnames=("cfg_fmm", "result", "G", "pout"))
fast_multipole_method.smap = shard_map_constructor(
    fast_multipole_method,
    in_specs=(P(-1), None, P(-1), None, None, None),
    static_argnames=("cfg_fmm", "result", "G", "pout"),
)
