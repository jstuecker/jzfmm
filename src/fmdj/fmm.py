from typing import Tuple, List
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import warnings
from dataclasses import dataclass, field, replace


from jztree.data import PosMass, InteractionList, get_pos_mass, TreeHierarchy, PosLvl, PackedArray, RankIdx, get_num
from jztree.tree import grouped_dense_interaction_list, zsort_and_tree
from jztree.tree import distr_grouped_dense_interaction_list, simplify_interaction_list
from jztree.jax_ext import pcast_like, raise_if, shard_map_constructor
from jztree.tools import masked_inverse
from jztree.comm import all_to_all_request_children, all_to_all_with_irank, get_rank_info, in_shard_map_context
from jax.sharding import PartitionSpec as P

from .config import Config
from .data import LocalExpansion
from .multipoles import _fmm_node_to_child, build_multipole_hierarchy, local_readout_pos_vjp, num_multi, p_of_num_multi, shift_local_to_children_vjp_x

import fmdj_cuda.ffi_fmm as ffi_fmm
import fmdj_cuda.ffi_pair_summation as ffi_pair_summation
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

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FMMNodeData:
    cent: jax.Array
    loc: jax.Array
    num: jax.Array

def _fmm_node_to_node(
        node_range: jax.Array,
        node_ilist: InteractionList,
        spl_recv: jax.Array,
        child_recv: FMMChildData,
        cfg: Config,
        spl_src: jax.Array | None = None,
        child_src: FMMChildData | None = None
    ) -> Tuple[jax.Array, InteractionList]:
    """
    Evaluates M2L for a tree plane and generates child interaction list.

    Outputs:
    - loc: multipole local expansion, shape (Nchild, M)
    - new_ilist: new interaction list for children
    """
    if spl_src is None:
        spl_src = spl_recv
    if child_src is None:
        child_src = child_recv

    size = len(child_recv.poslvl.pos)
    ilist_alloc_size = cfg.fmm.ilist_alloc_fac * size
    kernel = cfg.kernel
    dim = child_recv.poslvl.pos.shape[-1]

    if cfg.fmm.p == 5 and cfg.fmm.p_extra_m2l == 1 and dim == 3:
        warnings.warn(
            "FMMConfig(p=5, p_extra_m2l=1) currently makes the CUDA M2L kernel use local memory "
            "and will be extremely slow. Please prefer other p+p_extra_m2l <= 5.",
            RuntimeWarning,
            stacklevel=2,
        )

    assert ilist_alloc_size < 2**31, "So far only int32 supported {ilist_alloc_size/2**31}"
    assert len(spl_recv) == len(node_ilist.ispl)

    # children = jnp.concatenate((plane.center(), plane.lvl.view(jnp.float32)[...,None]), axis=-1)
    children_recv = child_recv.poslvl.pos_lvl()
    children_src = child_src.poslvl.pos_lvl()
    
    # Determine output shapes
    dtype = child_src.mp.dtype
    kernel_params = kernel.params(dtype=dtype)
    opening = cfg.fmm.opening
    opening_params = opening.params(dtype=dtype)
    p_local = cfg.fmm.p + cfg.fmm.p_extra_m2l
    assert p_local >= 0, "p + p_extra_m2l must be non-negative"
    out_loc = jax.ShapeDtypeStruct((size, num_multi(p_local, dim=dim)), dtype)
    out_interaction_count = jax.ShapeDtypeStruct((size,), jnp.int32)
    
    # Count opened interactions and evaluate M2L
    loc, interaction_counts = jax.ffi.ffi_call(
        "CountInteractionsAndM2L",
        (out_loc, out_interaction_count, )
    )(
        node_range, spl_recv, spl_src, node_ilist.ispl, node_ilist.isrc,
        children_recv, children_src, child_src.mp, kernel_params, opening_params,
        p=np.int32(cfg.fmm.p),
        p_extra_m2l=np.int32(cfg.fmm.p_extra_m2l),
        radial_kernel_kind=np.int32(kernel.kind_id()),
        opening_criterion_kind=np.int32(opening.kind_id()),
    )
    loc, interaction_counts = jax.tree.map(
        lambda x: pcast_like(x, spl_recv), (loc, interaction_counts)
    )

    # Insert interactions
    ispl_child = jnp.pad(jnp.cumsum(interaction_counts), (1, 0))
    out_child_ilist = jax.ShapeDtypeStruct((ilist_alloc_size,), jnp.int32)

    child_ilist = jax.ffi.ffi_call(
        "InsertInteractions",
        (out_child_ilist,)
    )(
        node_range, spl_recv, spl_src, node_ilist.ispl, node_ilist.isrc,
        children_recv, children_src, ispl_child, opening_params,
        opening_criterion_kind=np.int32(opening.kind_id()),
    )[0]
    child_ilist = pcast_like(child_ilist, spl_recv)

    # Create interaction list from outputs
    new_ilist = InteractionList(ispl=ispl_child, isrc=child_ilist)

    new_ilist.ispl = new_ilist.ispl + raise_if(
        new_ilist.nfilled() > new_ilist.size(), 
        "Interaction list allocation too small ({nfil}/{size})\nHint: Increase alloc_fac_ilist", 
        nfil=new_ilist.nfilled(), size=new_ilist.size()
    )
    
    return loc, new_ilist
_fmm_node_to_node.jit = jax.jit(_fmm_node_to_node, static_argnames=['cfg'])

def _fmm_dual_walk(th: TreeHierarchy, mph: PackedArray, cfg: Config):
    in_smap = in_shard_map_context()
    if in_smap:
        rank, ndev, axis_name = get_rank_info()

    # define root level:
    size = th.size()
    center0 = th.center().get(0, size)
    dim = center0.shape[-1]
    if in_smap:
        spl, ilist, nsup = distr_grouped_dense_interaction_list(
            th.num(th.num_planes()-1), size, size_ilist=int(size*cfg.fmm.ilist_alloc_fac)
        )
    else:
        spl, ilist, nsup = grouped_dense_interaction_list(
            th.num(th.num_planes()-1), size_ilist=int(size*cfg.fmm.ilist_alloc_fac),
            ngroup=32, size_super=size
        )

    p_local = cfg.fmm.p + cfg.fmm.p_extra_m2l
    assert p_local >= 0, "p + p_extra_m2l must be non-negative"
    loc = jnp.zeros((size, num_multi(p_local, dim=dim)), dtype=mph.data.dtype)
    if in_smap:
        loc = pcast_like(loc, mph.data)

    # Add super node data into tree information
    spl_n2n = th.ispl_n2n.append(spl, nsup+1, fill_value=spl[-1], resize=True)
    cent = th.center().append(jnp.zeros((size, dim), dtype=center0.dtype), nsup, fill_value=0., resize=True)

    def handle_level(i, carry):
        level = th.num_planes() - 1 - i

        parent_loc, parent_ilist = carry
        parent_spl_recv = spl_n2n.get(level+1, size+1)
        parent_cent = cent.get(level+1, size)

        child_recv = FMMChildData(
            poslvl=PosLvl(pos=cent.get(level, size), lvl=th.lvl.get(level, size)),
            mp=mph.get(level, size=size)
        )

        if in_smap:
            # Request the remote source node data that local receivers interact with.
            # This also runs in size-one shard maps as a syntax/shape test path.
            (child_src, ids), parent_spl_src, dev_spl = all_to_all_request_children(
                parent_ilist.dev_spl, parent_ilist.ids, parent_spl_recv,
                (child_recv, jnp.arange(size)),
                axis_name=axis_name, err_hint_child="\nHint: increase alloc_fac_nodes",
                err_hint_parent="\nHint: increase alloc_fac_nodes"
            )
            parent_range = jnp.array([0, parent_ilist.ispl.size-1], dtype=jnp.int32)
        else:
            child_src, parent_spl_src = child_recv, parent_spl_recv
            parent_range = jnp.array([0, spl_n2n.num(level+1)-1], dtype=jnp.int32)

        loc_ch, ilist = _fmm_node_to_node(
            parent_range, parent_ilist, parent_spl_recv, child_recv,
            cfg=cfg, spl_src=parent_spl_src, child_src=child_src
        )

        loc_ch = loc_ch + _fmm_node_to_child(
            parent_spl_recv, parent_loc, parent_cent, child_recv.poslvl.pos, cfg=cfg
        )

        if in_smap:
            ilist = replace(ilist, ids=ids, dev_spl=dev_spl)
            ilist = simplify_interaction_list(ilist)

        return loc_ch, ilist

    loc, ilist = jax.lax.fori_loop(
        0, th.num_planes(), handle_level, (loc, ilist), unroll=True
    )

    return loc, ilist
_fmm_dual_walk.jit = jax.jit(_fmm_dual_walk, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                           Leaf-Leaf (Particle to Particle) Interactions                          #
# ------------------------------------------------------------------------------------------------ #

def leaf_leaf_summation(
        particles: PosMass,
        ispl: jax.Array,
        ilist: InteractionList,
        cfg: Config = None,
    ) -> jax.Array:
    particles_recv = particles
    spl_recv = ispl
    in_smap = in_shard_map_context()
    if in_smap:
        rank, _, axis_name = get_rank_info()

    block_size = 128
    assert cfg.tree.max_leaf_size <= block_size
    dim = particles_recv.pos.shape[-1]

    node_range = jnp.array([0, spl_recv.size-1], dtype=jnp.int32)
    posm_recv = get_pos_mass(particles_recv)
    out_type = jax.ShapeDtypeStruct((particles_recv.pos.shape[0], dim + 1), posm_recv.dtype)
    kernel = cfg.kernel
    kernel_params = kernel.params(dtype=posm_recv.dtype)

    def eval_fwd(particles_recv, spl_recv, ilist):
        if in_smap:
            particles_src, spl_src, _dev_spl_src = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, spl_recv, particles_recv,
                axis_name=axis_name,
                err_hint_parent="\nHint: increase alloc_fac_nodes.",
                err_hint_child="\nHint: increase padding."
            )
            ilist = ilist.without_remote_query_points(rank)
        else:
            particles_src, spl_src = particles_recv, spl_recv

        loc = jax.ffi.ffi_call("LeafLeafPairSummation", (out_type,))(
            node_range, spl_recv, spl_src, ilist.ispl, ilist.isrc,
            get_pos_mass(particles_recv), get_pos_mass(particles_src), kernel_params,
            radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
            kahan=bool(cfg.fmm.kahan_summation)
        )[0]
        loc = pcast_like(loc, spl_recv)
        loc = loc.at[...,0].add(-particles_recv.mass*kernel.self_value()) # Remove self-interaction from potential
        num = getattr(particles_recv, "num", None)
        if num is not None:
            valid = jnp.arange(loc.shape[0]) < num
            loc = jnp.where(valid[:, None], loc, jnp.nan)
        return loc, (particles_recv, spl_recv, ilist, particles_src, spl_src)

    @jax.custom_vjp
    def eval(particles_recv, spl_recv, ilist):
        return eval_fwd(particles_recv, spl_recv, ilist)[0]
    
    def eval_bwd(res, gloc):
        particles_recv, spl_recv, ilist, particles_src, spl_src = res
        if in_smap:
            gloc_src, _, _ = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, spl_recv, gloc,
                output=jnp.zeros((particles_src.pos.shape[0], gloc.shape[1]), dtype=gloc.dtype),
                axis_name=axis_name,
                err_hint_parent="\nHint: increase alloc_fac_nodes",
                err_hint_child="\nHint: increase padding."
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
            kahan=bool(cfg.fmm.kahan_summation)
        )[0]
        gpos = pcast_like(gposm[:,:dim], particles_recv.pos)
        gmass = pcast_like(gposm[:,dim], particles_recv.mass)
        gnum = None
        if particles_recv.num is not None:
            gnum = jnp.zeros_like(particles_recv.num, dtype=jax.dtypes.float0)

        gposm = PosMass(
            pos=gpos, mass=gmass, num=gnum, num_total=particles_recv.num_total
        )
        return gposm, None, None
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(particles_recv, spl_recv, ilist)
leaf_leaf_summation.jit = jax.jit(leaf_leaf_summation, static_argnames=['cfg'])


# ------------------------------------------------------------------------------------------------ #
#                                      Direct Summation Forces                                     #
# ------------------------------------------------------------------------------------------------ #

def direct_summation(part: PosMass, kahan: bool = False,
                     kernel = None, G = 1
                     ) -> LocalExpansion:
    block_size = 64
    posm = get_pos_mass(part)
    out_type = jax.ShapeDtypeStruct(posm.shape, posm.dtype)
    if kernel is None:
        from .config import PlummerKernel
        kernel = PlummerKernel()
    kernel_params = kernel.params(dtype=posm.dtype)
    
    @jax.custom_vjp
    def eval(xm):
        loc = jax.ffi.ffi_call("DirectPairSummation", (out_type,))(
        xm, kernel_params, block_size=np.uint64(block_size),
        radial_kernel_kind=np.int32(kernel.kind_id()), kahan=kahan)[0]
        return loc
    def eval_fwd(xm):
        return eval(xm), xm
    def eval_bwd(xm, gloc):
        gxm = jax.ffi.ffi_call("BwdDirectPairSummation", (out_type,))(
            gloc, xm, kernel_params, block_size=np.uint64(block_size),
            radial_kernel_kind=np.int32(kernel.kind_id()), kahan=kahan
        )[0]
        return gxm,
    
    eval.defvjp(eval_fwd, eval_bwd)

    return LocalExpansion(eval(posm) * G, dim=part.pos.shape[-1])
direct_summation.jit = jax.jit(direct_summation, static_argnames=['kahan', 'kernel'])

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

def direct_force_and_potential_jax(x, m=1., softening=1e-2):
    phi = direct_potential_jax(x, m, softening)
    f = direct_force_jax(x, m, softening)
    
    return jnp.concatenate([f, phi[:,None]], axis=-1)
direct_force_and_potential_jax.jit = jax.jit(direct_force_and_potential_jax)

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


def evaluate_node_node_fmm(partz: PosMass, th: TreeHierarchy, *, cfg: Config) -> Tuple[jax.Array, InteractionList]:
    
    def eval_fwd(pos, mp, pout=1):
        mph = build_multipole_hierarchy(th, pos, mp, cfg=cfg)
        loc_node, ilist = _fmm_dual_walk(th, mph, cfg=cfg)
        ispl = th.splits_leaf_to_part()
        xnode = th.center().get(0, th.size())
        loc_part = _fmm_node_to_child(ispl, loc_node, xnode, pos, pout=pout, cfg=cfg)
        return (loc_part, ilist), (pos, mp, ispl, xnode, loc_node)
    
    def eval_bwd(pout, res, grads):
        assert cfg.fmm.p_extra_m2l == 0, (
            "Differentiating through p_extra_m2l != 0 is not supported yet. "
            "The forward pass supports boosted M2L locals, but the rectangular M2L adjoint still needs to be added."
        )
        pos, mp, ispl, xnode, loc_node = res
        gloc = grads[0]

        # Backwards pass = FMM with gloc as multipole weights
        # plus the position derivatives of the shifting operators

        gx1 = shift_local_to_children_vjp_x(ispl, loc_node, xnode, pos, gloc)
        
        dim = pos.shape[-1]
        (gmp, _), (_, _, _, _, gmp_node) = eval_fwd(pos, gloc, pout=p_of_num_multi(mp.shape[-1], dim=dim))

        gx2 = shift_local_to_children_vjp_x(ispl, gmp_node, xnode, pos, mp)
        
        return gx1 + gx2, gmp
    
    @partial(jax.custom_vjp, nondiff_argnames=['pout'])
    def eval(pos, mp, pout=1):
        return eval_fwd(pos, mp, pout=pout)[0]
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(partz.pos, jnp.reshape(partz.mass, jnp.shape(partz.mass) + (1,)))
evaluate_node_node_fmm.jit = jax.jit(evaluate_node_node_fmm, static_argnames=['cfg', ])

def _fast_multipole_method_z(partz: PosMass, th: TreeHierarchy, *, cfg: Config, pout: int = 1) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."

    loc_node_node, ilist = evaluate_node_node_fmm(partz, th, cfg=cfg)
    spl = th.splits_leaf_to_part()

    loc_leaf_leaf = leaf_leaf_summation(partz, spl, jax.lax.stop_gradient(ilist), cfg=cfg)
    loc = loc_leaf_leaf + loc_node_node

    return LocalExpansion(loc * cfg.G(), dim=partz.pos.shape[-1])
_fast_multipole_method_z.jit = jax.jit(_fast_multipole_method_z, static_argnames=("cfg", "pout"))

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
        part: PosMass, cfg: Config, th: TreeHierarchy | None = None,
        result: str = "loc", pout: int = 1
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
            part, cfg.tree, data=origin
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
        locz = _fast_multipole_method_z(_as_posmass(partz), th, cfg=cfg, pout=pout)

    loc = None
    if "loc" in keys:
        if in_smap:
            (loc_values, idx), _dev_spl = all_to_all_with_irank(
                origin_z.rank, (locz.values, origin_z.idx), num=partz.num,
                axis_name=axis_name, err_hint="\nThis should never fail..."
            )
        else:
            loc_values, idx = locz.values, origin_z.idx
        inv_sort = masked_inverse(idx, mask=jnp.arange(len(idx)) < num_origin)
        loc = LocalExpansion(loc_values[inv_sort], dim=part.pos.shape[-1])

    out = {
        "loc": loc,
        "locz": locz,
        "partz": partz,
        "tree": th,
    }
    res = tuple(out[key] for key in keys)
    return res[0] if len(res) == 1 else res
fast_multipole_method.jit = jax.jit(fast_multipole_method, static_argnames=("cfg", "result", "pout"))
fast_multipole_method.smap = shard_map_constructor(
    fast_multipole_method,
    in_specs=(P(-1), None, P(-1), None, None),
    static_argnames=("cfg", "result", "pout"),
)

def force_and_potential(p: PosMass, cfg : Config) -> LocalExpansion:
    if cfg.fmm is None:
        return direct_summation(p, kernel=cfg.kernel, kahan=True, G=cfg.G())
    else:
        return fast_multipole_method(p, cfg=cfg, pout=1)
force_and_potential.jit = jax.jit(force_and_potential, static_argnames=("cfg",))
