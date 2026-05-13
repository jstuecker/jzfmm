from typing import Tuple, List
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


from jztree.data import PosMass, InteractionList, get_pos_mass, TreeHierarchy, PosLvl, PackedArray
from jztree.tree import zsort, build_tree_hierarchy, grouped_dense_interaction_list
from jztree.jax_ext import raise_if

from .config import Config
from .data import LocalExpansion
from .multipoles import _fmm_node_to_child, build_multipole_hierarchy, local_readout_pos_vjp, p_of_num_multi, shift_local_to_children_vjp_x

import fmdj_cuda.ffi_fmm as ffi_fmm
import fmdj_cuda.ffi_forces as ffi_forces
jax.ffi.register_ffi_target("CountInteractionsAndM2L", ffi_fmm.CountInteractionsAndM2L(), platform="CUDA")
jax.ffi.register_ffi_target("InsertInteractions", ffi_fmm.InsertInteractions(), platform="CUDA")
jax.ffi.register_ffi_target("GroupedForceAndPot", ffi_forces.GroupedForceAndPot(), platform="CUDA")
jax.ffi.register_ffi_target("BwdGroupedForceAndPot", ffi_forces.BwdGroupedForceAndPot(), platform="CUDA")
jax.ffi.register_ffi_target("ForceAndPotential", ffi_forces.ForceAndPotential(), platform="CUDA")
jax.ffi.register_ffi_target("BwdForceAndPotential", ffi_forces.BwdForceAndPotential(), platform="CUDA")

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
        spl: jax.Array,
        child_data: FMMChildData,
        cfg: Config
    ) -> Tuple[jax.Array, InteractionList]:
    """
    Evaluates M2L for a tree plane and generates child interaction list.

    Outputs:
    - loc: multipole local expansion, shape (Nchild, M)
    - new_ilist: new interaction list for children
    """
    size = len(child_data.poslvl.pos)
    ilist_alloc_size = cfg.fmm.ilist_alloc_fac * size
    kernel = cfg.kernel

    assert ilist_alloc_size < 2**31, "So far only int32 supported {ilist_alloc_size/2**31}"
    assert len(spl) == len(node_ilist.ispl)

    # children = jnp.concatenate((plane.center(), plane.lvl.view(jnp.float32)[...,None]), axis=-1)
    children = child_data.poslvl.pos_lvl()
    
    # Determine output shapes
    out_loc = jax.ShapeDtypeStruct(child_data.mp.shape, jnp.float32)
    out_interaction_count = jax.ShapeDtypeStruct((size,), jnp.int32)
    
    # Count opened interactions and evaluate M2L
    loc, interaction_counts = jax.ffi.ffi_call(
        "CountInteractionsAndM2L",
        (out_loc, out_interaction_count, )
    )(
        node_range, spl, node_ilist.ispl, node_ilist.isrc, children, child_data.mp, kernel.params(),
        p=np.int32(cfg.fmm.p),
        radial_kernel_kind=np.int32(kernel.kind_id()),
        opening_angle=np.float32(cfg.fmm.opening_angle)
    )

    # Insert interactions
    ispl_child = jnp.pad(jnp.cumsum(interaction_counts), (1, 0))
    out_child_ilist = jax.ShapeDtypeStruct((ilist_alloc_size,), jnp.int32)

    child_ilist = jax.ffi.ffi_call(
        "InsertInteractions",
        (out_child_ilist,)
    )(
        node_range, spl, node_ilist.ispl, node_ilist.isrc, children, ispl_child,
        opening_angle=np.float32(cfg.fmm.opening_angle)
    )[0]

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
    # define root level:
    size = th.size()
    dim = th.center().get(0, size).shape[-1]
    spl, ilist, nsup = grouped_dense_interaction_list(
        th.num(th.num_planes()-1), size_ilist=int(size*cfg.fmm.ilist_alloc_fac), ngroup=32, size_super=size
    )

    loc = jnp.zeros((size, mph.data.shape[-1]), dtype=jnp.float32)

    # Add super node data into tree information
    spl_n2n = th.ispl_n2n.append(spl, nsup+1, fill_value=spl[-1], resize=True)
    cent = th.center().append(jnp.zeros((size, dim), dtype=jnp.float32), nsup, fill_value=0., resize=True)

    def handle_level(i, carry):
        level = th.num_planes() - 1 - i

        parent_loc, parent_ilist = carry
        parent_spl = spl_n2n.get(level+1, size+1)

        parent_range = jnp.array([0, spl_n2n.num(level+1)-1], dtype=jnp.int32)
        child_data = FMMChildData(
            poslvl=PosLvl(pos=cent.get(level, size), lvl=th.lvl.get(level, size)),
            mp=mph.get(level, size=size)
        )

        loc_ch, ilist = _fmm_node_to_node(
            parent_range, parent_ilist, parent_spl, child_data, cfg=cfg
        )

        loc_ch = loc_ch + _fmm_node_to_child(
            parent_spl, parent_loc, cent.get(level+1, size), cent.get(level, size), cfg=cfg
        )

        return loc_ch, ilist

    # unrolling the loop turns out better, since number of iterations tends to be very small
    loc, ilist = jax.lax.fori_loop(
        0, th.num_planes(), handle_level, (loc, ilist), unroll=True
    )

    return loc, ilist
_fmm_dual_walk.jit = jax.jit(_fmm_dual_walk, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                           Leaf-Leaf (Particle to Particle) Interactions                          #
# ------------------------------------------------------------------------------------------------ #

def grouped_force_and_pot(particles: PosMass, ispl: jax.Array, ilist: InteractionList,
                          cfg: Config = None) -> jax.Array:
    block_size = 128
    assert cfg.tree.max_leaf_size <= block_size
    assert len(ispl) == len(ilist.ispl)
    dim = particles.pos.shape[-1]

    node_range = jnp.array([0, ispl.size-1], dtype=jnp.int32)
    out_type = jax.ShapeDtypeStruct((particles.pos.shape[0], dim + 1), jnp.float32)
    kernel = cfg.kernel

    @jax.custom_vjp
    def eval(particles, ispl, ilist):
        loc = jax.ffi.ffi_call("GroupedForceAndPot", (out_type,))(
            node_range, ispl, ilist.ispl, ilist.isrc, get_pos_mass(particles), kernel.params(),
            radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
            kahan=bool(cfg.fmm.kahan_summation)
        )[0]
        loc = loc.at[...,0].add(particles.mass*kernel.self_value()) # Remove self-interaction from potential
        return loc
    
    def eval_fwd(particles, ispl, ilist):
        return eval(particles, ispl, ilist), (particles, ispl, ilist)
    
    def eval_bwd(res, gloc):
        particles, ispl, ilist = res
        gposm = jax.ffi.ffi_call("BwdGroupedForceAndPot", (out_type,))(
            node_range, ispl, ilist.ispl, ilist.isrc, get_pos_mass(particles), kernel.params(), gloc,
            radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
            kahan=bool(cfg.fmm.kahan_summation)
        )[0]
        return PosMass(pos=gposm[:,:dim], mass=gposm[:,dim]), None, None
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(particles, ispl, ilist)
grouped_force_and_pot.jit = jax.jit(grouped_force_and_pot, static_argnames=['cfg'])


# ------------------------------------------------------------------------------------------------ #
#                                      Direct Summation Forces                                     #
# ------------------------------------------------------------------------------------------------ #

def direct_force_and_potential(part: PosMass, kahan: bool = False,
                               kernel = None
                               ) -> jax.Array:
    block_size = 64
    posm = get_pos_mass(part)
    out_type = jax.ShapeDtypeStruct(posm.shape, posm.dtype)
    if kernel is None:
        from .config import PlummerKernel
        kernel = PlummerKernel()
    
    @jax.custom_vjp
    def eval(xm):
        loc = jax.ffi.ffi_call("ForceAndPotential", (out_type,))(
        xm, kernel.params(), block_size=np.uint64(block_size),
        radial_kernel_kind=np.int32(kernel.kind_id()), kahan=kahan)[0]
        return loc
    def eval_fwd(xm):
        return eval(xm), xm
    def eval_bwd(xm, gloc):
        gxm = jax.ffi.ffi_call("BwdForceAndPotential", (out_type,))(
            gloc, xm, kernel.params(), block_size=np.uint64(block_size),
            radial_kernel_kind=np.int32(kernel.kind_id()), kahan=kahan
        )[0]
        return gxm,
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(posm)
direct_force_and_potential.jit = jax.jit(direct_force_and_potential, static_argnames=['kahan', 'kernel'])

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

def fast_multipole_method_z(partz: PosMass, *, mpz: jax.Array | None = None, cfg: Config, pout: int = 1) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."

    if mpz is None:
        mpz = partz.mass

    th = build_tree_hierarchy(jax.lax.stop_gradient(partz), cfg.tree)

    loc_node_node, ilist = evaluate_node_node_fmm(partz, th, cfg=cfg)
    spl = th.splits_leaf_to_part()

    loc_leaf_leaf = grouped_force_and_pot(partz, spl, jax.lax.stop_gradient(ilist), cfg=cfg)
    loc = loc_leaf_leaf + loc_node_node

    return LocalExpansion(loc * cfg.G(), dim=partz.pos.shape[-1])
fast_multipole_method_z.jit = jax.jit(fast_multipole_method_z, static_argnames=("cfg", "pout"))

def fast_multipole_method(part: PosMass, *, cfg: Config, pout: int = 1) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."

    partz, isortz = zsort(PosMass(pos=part.pos, mass=part.mass))

    locz = fast_multipole_method_z(partz, cfg=cfg, pout=pout)

    inv_sort = jnp.zeros_like(isortz).at[isortz].set(jnp.arange(len(isortz), dtype=isortz.dtype))

    return LocalExpansion(locz.values[inv_sort], dim=part.pos.shape[-1])
fast_multipole_method.jit = jax.jit(fast_multipole_method, static_argnames=("cfg", "pout"))

def force_and_potential(p: PosMass, cfg : Config) -> LocalExpansion:
    if cfg.fmm is None:
        loc = direct_force_and_potential(p, kernel=cfg.kernel, kahan=True) * cfg.G()
        return LocalExpansion(loc, dim=p.pos.shape[-1])
    else:
        return fast_multipole_method(p, cfg=cfg, pout=1)
force_and_potential.jit = jax.jit(force_and_potential, static_argnames=("cfg",))
