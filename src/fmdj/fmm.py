from typing import Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from .config import Config
from .data import TreePlane, PosMass, InteractionList, LocalExpansion
from .ztree import pos_zorder_sort, build_tree_hierarchy, grouped_dense_interaction_list
from .multipoles import shift_local_to_children, build_multipole_hierarchy, local_readout_pos_vjp, p_of_num_multi, shift_local_to_children_vjp_x

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

def evaluate_plane_interactions(
        plane: TreePlane,
        mp: jax.Array,
        plane_lr: TreePlane | None = None,
        ilist_lr: InteractionList | None = None,
        loc_lr: jax.Array | None = None,
        cfg: Config = None
    ) -> Tuple[jax.Array, InteractionList]:
    """
    Evaluates M2L for a tree plane and generates child interaction list.
    
    Inputs:
    - plane: TreePlane at current level
    - plane_lr: TreePlane at lower resolution (parent level), optional
    - ilist_lr: InteractionList at lower resolution, optional
    - loc_lr: Local expansion at lower resolution, optional
    - cfg: Configuration object
    
    Outputs:
    - loc: multipole local expansion, shape (Nchild, M)
    - new_ilist: new interaction list for children
    """
    ilist_alloc_size = cfg.fmm.ilist_alloc_fac * plane.size()

    assert ilist_alloc_size < 2**31, "So far only int32 supported {ilist_alloc_size/2**31}"

    if plane_lr is None or ilist_lr is None: # Root level
        spl_nodes, ilist_lr, nnodes = grouped_dense_interaction_list(plane.nnodes, ilist_alloc_size, ngroup=8)
        node_range = jnp.array([0, nnodes], dtype=jnp.int32)
    else:
        spl_nodes = plane_lr.ispl
        node_range = jnp.array([0, plane_lr.nnodes], dtype=jnp.int32)

    children = jnp.concatenate((plane.center(), plane.lvl.view(jnp.float32)[...,None]), axis=-1)
    
    # Determine output shapes
    out_loc = jax.ShapeDtypeStruct(mp.shape, jnp.float32)
    out_interaction_count = jax.ShapeDtypeStruct((plane.size(),), jnp.int32)
    
    # Count opened interactions and evaluate M2L
    loc, interaction_counts = jax.ffi.ffi_call(
        "CountInteractionsAndM2L",
        (out_loc, out_interaction_count, )
    )(
        node_range, spl_nodes, ilist_lr.ispl, ilist_lr.iother, children, mp,
        p=np.int32(cfg.fmm.p),
        softening=np.float32(cfg.softening),
        opening_angle=np.float32(cfg.fmm.opening_angle)
    )

    # Insert interactions
    ispl_child = jnp.pad(jnp.cumsum(interaction_counts), (1, 0))
    out_child_ilist = jax.ShapeDtypeStruct((ilist_alloc_size,), jnp.int32)

    child_ilist = jax.ffi.ffi_call(
        "InsertInteractions",
        (out_child_ilist,)
    )(
        node_range, spl_nodes, ilist_lr.ispl, ilist_lr.iother, children, ispl_child,
        opening_angle=np.float32(cfg.fmm.opening_angle)
    )[0]

    # Create interaction list from outputs
    new_ilist = InteractionList(ispl=ispl_child, iother=child_ilist, nfilled=ispl_child[-1])

    # Evaluate L2L part
    if loc_lr is not None:
        loc = loc + shift_local_to_children(
            plane_lr.ispl, loc_lr, plane_lr.center(), plane.center(), cfg=cfg
        )
    
    return loc, new_ilist
evaluate_plane_interactions.jit = jax.jit(evaluate_plane_interactions, static_argnames=['cfg'])

def evaluate_interaction_hierarchy(th, mph, cfg):
    ilist, loc, last_plane = None, None, None
    for i in reversed(range(0, len(th))):
        loc, ilist = evaluate_plane_interactions(th[i], mph[i], last_plane, ilist, loc, cfg=cfg)
        last_plane = th[i]
    return loc, ilist
evaluate_interaction_hierarchy.jit = jax.jit(evaluate_interaction_hierarchy, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                           Leaf-Leaf (Particle to Particle) Interactions                          #
# ------------------------------------------------------------------------------------------------ #

def grouped_force_and_pot(particles: PosMass, ispl: jax.Array, ilist: InteractionList,
                          cfg: Config = None) -> jax.Array:
    block_size = 128
    assert cfg.tree.max_leaf_size <= block_size
    node_range = jnp.array([0, ispl.size-1], dtype=jnp.int32)
    out_type = jax.ShapeDtypeStruct((particles.pos.shape[0], 4), jnp.float32)

    @jax.custom_vjp
    def eval(particles, ispl, ilist):
        loc = jax.ffi.ffi_call("GroupedForceAndPot", (out_type,))(
            node_range, ispl, ilist.ispl, ilist.iother, particles.posm(),
            softening=np.float32(cfg.softening), block_size=np.uint64(block_size),
            kahan=bool(cfg.fmm.kahan_summation)
        )[0]
        loc = loc.at[...,0].add(particles.mass/cfg.softening) # Remove self-interaction from potential
        return loc
    
    def eval_fwd(particles, ispl, ilist):
        return eval(particles, ispl, ilist), (particles, ispl, ilist)
    
    def eval_bwd(res, gloc):
        particles, ispl, ilist = res
        gposm = jax.ffi.ffi_call("BwdGroupedForceAndPot", (out_type,))(
            node_range, ispl, ilist.ispl, ilist.iother, particles.posm(), gloc,
            softening=np.float32(cfg.softening), block_size=np.uint64(block_size),
            kahan=bool(cfg.fmm.kahan_summation)
        )[0]
        return PosMass(gposm[:,0:3], gposm[:,3]), None, None
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(particles, ispl, ilist)
grouped_force_and_pot.jit = jax.jit(grouped_force_and_pot, static_argnames=['cfg'])


# ------------------------------------------------------------------------------------------------ #
#                                      Direct Summation Forces                                     #
# ------------------------------------------------------------------------------------------------ #

def direct_force_and_potential(posm: PosMass, softening: float = 1e-2, kahan: bool = False
                               ) -> jax.Array:
    block_size = 64
    out_type = jax.ShapeDtypeStruct(posm.posm().shape, posm.posm().dtype)
    
    @jax.custom_vjp
    def eval(xm):
        loc = jax.ffi.ffi_call("ForceAndPotential", (out_type,))(
        xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), kahan=kahan)[0]
        return loc
    def eval_fwd(xm):
        return eval(xm), xm
    def eval_bwd(xm, gloc):
        gxm = jax.ffi.ffi_call("BwdForceAndPotential", (out_type,))(
            gloc, xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), kahan=kahan
        )[0]
        return gxm,
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(posm.posm())
direct_force_and_potential.jit = jax.jit(direct_force_and_potential, static_argnames=['softening', 'kahan'])

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


def evaluate_node_node_fmm(partz: PosMass, th: list[TreePlane], *, cfg: Config) -> Tuple[jax.Array, InteractionList]:
    
    def eval_fwd(pos, mp, pout=1):
        mph = build_multipole_hierarchy(th, pos, mp, cfg=cfg)
        loc_node, ilist = evaluate_interaction_hierarchy(th, mph, cfg=cfg)
        loc_part = shift_local_to_children(th[0].ispl, loc_node, th[0].center(), pos, pout=pout, cfg=cfg)
        return (loc_part, ilist), (pos, mp, th, loc_node)
    
    def eval_bwd(pout, res, grads):
        pos, mp, th, loc_node = res
        gloc = grads[0]

        # Backwards pass = FMM with gloc as multipole weights
        # plus the position derivatives of the shifting operators

        gx1 = shift_local_to_children_vjp_x(th[0].ispl, loc_node, th[0].center(), pos, gloc)
        
        (gmp, _), (_, _, _, gmp_node) = eval_fwd(pos, gloc, pout=p_of_num_multi(mp.shape[-1]))

        gx2 = shift_local_to_children_vjp_x(th[0].ispl, gmp_node, th[0].center(), pos, mp)
        
        return gx1 + gx2, gmp
    
    @partial(jax.custom_vjp, nondiff_argnames=['pout'])
    def eval(pos, mp, pout=1):
        return eval_fwd(pos, mp, pout=pout)[0]
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(partz.pos, partz.mass.reshape(-1,1))
evaluate_node_node_fmm.jit = jax.jit(evaluate_node_node_fmm, static_argnames=['cfg', ])

def fast_multipole_method_z(partz: PosMass, *, mpz: jax.Array | None = None, cfg: Config, pout: int = 1) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."

    if mpz is None:
        mpz = partz.mass

    th = build_tree_hierarchy(jax.lax.stop_gradient(partz), cfg.tree)
    tps = list(th.planes())

    loc_node_node, ilist = evaluate_node_node_fmm(partz, tps, cfg=cfg)

    loc_leaf_leaf = grouped_force_and_pot(partz, tps[0].ispl, jax.lax.stop_gradient(ilist), cfg=cfg)
    loc = loc_leaf_leaf + loc_node_node

    return LocalExpansion(loc * cfg.G())
fast_multipole_method_z.jit = jax.jit(fast_multipole_method_z, static_argnames=("cfg", "pout"))

def fast_multipole_method(part: PosMass, *, cfg: Config, pout: int = 1) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."

    posz, isortz = pos_zorder_sort(part.pos)
    partz = PosMass(pos=posz, mass=part.mass[isortz])

    locz = fast_multipole_method_z(partz, cfg=cfg, pout=pout)

    inv_sort = jnp.zeros_like(isortz).at[isortz].set(jnp.arange(len(isortz), dtype=isortz.dtype))

    return LocalExpansion(locz.values[inv_sort])
fast_multipole_method.jit = jax.jit(fast_multipole_method, static_argnames=("cfg", "pout"))

def force_and_potential(p: PosMass, cfg : Config) -> LocalExpansion:
    if cfg.fmm is None:
        loc = direct_force_and_potential(p, softening=cfg.softening, kahan=True) * cfg.G()
        return LocalExpansion(loc)
    else:
        return fast_multipole_method(p, cfg=cfg, pout=1)
force_and_potential.jit = jax.jit(force_and_potential, static_argnames=("cfg",))