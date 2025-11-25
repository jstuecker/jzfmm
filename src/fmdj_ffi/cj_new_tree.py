import numpy as np

import jax
import jax.numpy as jnp

from fmdj.config import Config, FMMConfig
from fmdj.data import dense_interaction_list, TreePlane, PosMass, InteractionList
from fmdj.multipoles import shift_local_to_local

import fmdj_cuda.ffi_fmm as ffi_fmm
import fmdj_cuda.ffi_forces as ffi_forces

from typing import Tuple

jax.ffi.register_ffi_target("CountInteractionsAndM2L", ffi_fmm.CountInteractionsAndM2L(), platform="CUDA")
jax.ffi.register_ffi_target("InsertInteractions", ffi_fmm.InsertInteractions(), platform="CUDA")
jax.ffi.register_ffi_target("GroupedForceAndPot", ffi_forces.GroupedForceAndPot(), platform="CUDA")

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
    
    cfg_tree: FMMConfig = cfg.fmm
    nint_out = cfg_tree.ilist_alloc_fac * plane.size()

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
        p=np.int32(cfg_tree.p),
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

def cj_new_force_and_pot(particles: PosMass, 
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
cj_new_force_and_pot.jit = jax.jit(cj_new_force_and_pot, static_argnames=['cfg'])