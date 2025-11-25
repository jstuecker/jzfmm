from typing import Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from . import multipoles
from . import config
import fmdj.ztree

from .config import Config
from .data import TreePlane, PosMass, InteractionList, dense_interaction_list
from fmdj.ztree import build_tree_hierarchy
from fmdj.multipoles import shift_local_to_local

import fmdj_cuda.ffi_fmm as ffi_fmm
import fmdj_cuda.ffi_forces as ffi_forces

jax.ffi.register_ffi_target("CountInteractionsAndM2L", ffi_fmm.CountInteractionsAndM2L(), platform="CUDA")
jax.ffi.register_ffi_target("InsertInteractions", ffi_fmm.InsertInteractions(), platform="CUDA")
jax.ffi.register_ffi_target("GroupedForceAndPot", ffi_forces.GroupedForceAndPot(), platform="CUDA")
jax.ffi.register_ffi_target("ForceAndPotential", ffi_forces.ForceAndPotential(), platform="CUDA")
jax.ffi.register_ffi_target("BwdForceAndPotential", ffi_forces.BwdForceAndPotential(), platform="CUDA")


# ------------------------------------------------------------------------------------------------ #
#                                          M2L Evaluation                                          #
# ------------------------------------------------------------------------------------------------ #

def evaluate_plane_interactions(
        plane: TreePlane, 
        plane_lr: TreePlane | None = None,
        ilist_lr: InteractionList | None = None,
        loc_lr: jnp.ndarray | None = None,
        cfg: Config = None
    ) -> Tuple[jnp.ndarray, InteractionList]:
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
    if plane_lr is None or ilist_lr is None: # Root level
        ilist_lr = dense_interaction_list(plane.size(), nnodes=plane.nnodes)
        spl_nodes = jnp.minimum(jnp.arange(0, plane.size() + 1, dtype=jnp.int32), plane.nnodes)
        node_range = jnp.array([0, plane.nnodes], dtype=jnp.int32)
    else:
        spl_nodes = plane_lr.ispl
        node_range = jnp.array([0, plane_lr.nnodes], dtype=jnp.int32)
    
    nint_out = cfg.fmm.ilist_alloc_fac * plane.size()

    children = jnp.concatenate((plane.mp.center(), plane.lvl.view(jnp.float32)[...,None]), axis=-1)
    
    # Determine output shapes
    out_loc = jax.ShapeDtypeStruct(plane.mp.values.shape, jnp.float32)
    out_interaction_count = jax.ShapeDtypeStruct((plane.size(),), jnp.int32)
    
    # Count opened interactions and evaluate M2L
    loc, interaction_counts = jax.ffi.ffi_call(
        "CountInteractionsAndM2L",
        (out_loc, out_interaction_count, )
    )(
        node_range, spl_nodes, ilist_lr.ispl, ilist_lr.iother, children, plane.mp.values,
        p=np.int32(cfg.fmm.p),
        softening=np.float32(cfg.softening),
        opening_angle=np.float32(cfg.fmm.opening_angle)
    )

    # Insert interactions
    ispl_child = jnp.pad(jnp.cumsum(interaction_counts), (1, 0))
    out_child_ilist = jax.ShapeDtypeStruct((nint_out,), jnp.int32)

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
        ipar = plane_lr.icoarse_of_fine()
        loc = loc + shift_local_to_local(loc_lr[ipar], plane.mp.center() - plane_lr.mp.center()[ipar])
    
    return loc, new_ilist
evaluate_plane_interactions.jit = jax.jit(evaluate_plane_interactions, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                           Leaf-Leaf (Particle to Particle) Interactions                          #
# ------------------------------------------------------------------------------------------------ #

def grouped_force_and_pot(particles: PosMass, 
                         plane: TreePlane, 
                         ilist: InteractionList,
                         cfg: Config) -> jnp.ndarray:
    node_range = jnp.array([0, plane.nnodes], dtype=jnp.int32)
    
    fphi = jax.ffi.ffi_call("GroupedForceAndPot", (
        jax.ShapeDtypeStruct((particles.pos.shape[0], 4), jnp.float32),
    ))(
        node_range, plane.ispl, ilist.ispl, ilist.iother, particles.posm(),
        softening=np.float32(cfg.softening), max_leaf_size=np.int32(cfg.fmm.max_leaf_size),
        kahan=bool(cfg.fmm.kahan_summation)
    )[0]

    fphi = fphi.at[...,3].add(particles.mass/cfg.softening) # Remove self-interaction from potential

    return fphi
grouped_force_and_pot.jit = jax.jit(grouped_force_and_pot, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                                      Direct Summation Forces                                     #
# ------------------------------------------------------------------------------------------------ #

def direct_force_and_potential_fwd(xm, block_size=64, softening=1e-2, kahan=False):
    assert xm.dtype == jnp.float32
    assert xm.shape[-1] == 4
    assert xm.ndim >= 2
    
    assert softening > 0, "Epsilon must be positive to deal with self-interaction."

    out_type = jax.ShapeDtypeStruct(xm.shape, xm.dtype)
    fphi = jax.ffi.ffi_call("ForceAndPotential", (out_type,))(
        xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), kahan=kahan)[0]
    return fphi, xm

def force_and_potential_bwd(block_size, softening, kahan, xm, gfphi):
    out_type = jax.ShapeDtypeStruct(xm.shape, xm.dtype)
    gxm = jax.ffi.ffi_call("BwdForceAndPotential", (out_type,))(
        gfphi, xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), kahan=kahan)[0]
    return gxm,

@partial(jax.custom_vjp, nondiff_argnames=("block_size", "softening", "kahan"))
def direct_force_and_potential(xm, block_size=64, softening=1e-2, kahan=False):
    return direct_force_and_potential_fwd(xm, block_size, softening, kahan)[0]
direct_force_and_potential.defvjp(direct_force_and_potential_fwd, force_and_potential_bwd)
direct_force_and_potential.jit = jax.jit(direct_force_and_potential, static_argnames=("block_size", "softening", "kahan"))

def direct_potential_pure_jax(x, m=1., softening=1e-2):
    rij2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(rinv * jnp.broadcast_to(m, x.shape[:-1])[None,:], axis=1)
direct_potential_pure_jax.jit = jax.jit(direct_potential_pure_jax)

def direct_force_pure_jax(x, m=1., softening=1e-2):
    dx = x[:, None] - x[None, :]
    rij2 = jnp.sum(dx ** 2, axis=-1, keepdims=True)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(dx * rinv**3 * jnp.broadcast_to(m, x.shape[:-1])[None,:,None], axis=1)
direct_force_pure_jax.jit = jax.jit(direct_force_pure_jax)

def direct_force_and_potential_pure_jax(x, m=1., softening=1e-2):
    phi = direct_potential_pure_jax(x, m, softening)
    f = direct_force_pure_jax(x, m, softening)
    
    return jnp.concatenate([f, phi[:,None]], axis=-1)
direct_force_and_potential_pure_jax.jit = jax.jit(direct_force_and_potential_pure_jax)

# ------------------------------------------------------------------------------------------------ #
#                                         Master Functions                                         #
# ------------------------------------------------------------------------------------------------ #

def new_fmm_fphi(pos, mass, cfg : config.Config, return_sorted=False):
    if mass is None:
        mass = jnp.ones((pos.shape[0],), dtype=pos.dtype)
    elif jnp.shape(mass) != jnp.shape(pos)[:-1]:
        mass = jnp.broadcast_to(mass, pos.shape[:-1])

    posz, isortz = fmdj.ztree.pos_zorder_sort(pos)
    particlesz = PosMass(pos=posz, mass=mass[isortz])

    th = build_tree_hierarchy(particlesz, cfg)
    loc, ilist = evaluate_interaction_hierarchy(th, cfg=cfg)

    parent = th[0].icoarse_of_fine()
    fphi_loc = multipoles.evaluate_local_fphi(loc[parent], particlesz.pos - th[0].mp.center()[parent])

    fphi = grouped_force_and_pot(particlesz, th[0], ilist, cfg=cfg) + fphi_loc

    if return_sorted:
        return particlesz.pos, particlesz.mass, isortz, fphi
    else:
        fphi_unsorted = jnp.zeros_like(fphi).at[isortz].set(fphi)
        return fphi_unsorted
new_fmm_fphi.jit = jax.jit(new_fmm_fphi, static_argnames=("cfg", "return_sorted"))

def get_force_and_potential(pos, mass, cfg : config.Config, separately=False):
    if cfg.fmm is None: # Use direct summation
        xm = jnp.concatenate([pos, mass[:,None]], axis=-1)
        fphi = direct_force_and_potential(xm, softening=cfg.softening, kahan=True) * cfg.G()
    else:
        fphi = new_fmm_fphi(pos, mass, cfg=cfg) * cfg.G()

    if separately:
        return fphi[:,0:3], fphi[:,3]
    else:
        return fphi
get_force_and_potential.jit = jax.jit(get_force_and_potential, static_argnames=("cfg", "separately"))

def evaluate_interaction_hierarchy(th, cfg):
    ilist, loc, last_plane = None, None, None
    for i in reversed(range(0, len(th))):
        loc, ilist = evaluate_plane_interactions(th[i], last_plane, ilist, loc, cfg=cfg)
        last_plane = th[i]
    return loc, ilist
evaluate_interaction_hierarchy.jit = jax.jit(evaluate_interaction_hierarchy, static_argnames=['cfg'])