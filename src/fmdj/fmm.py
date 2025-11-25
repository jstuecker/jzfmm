import jax
import jax.numpy as jnp
from . import multipoles
from . import config
import fmdj_ffi as cj
import fmdj_ffi.cj_new_tree as cnt
import fmdj.ztree

from .config import Config
from .data import TreePlane, PosMass, InteractionList

from fmdj_ffi.cj_new_tree import evaluate_plane_interactions
from fmdj.multipoles import coarsen_multipoles, multipoles_from_particles


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

    fphi = cnt.grouped_force_and_pot(particlesz, th[0], ilist, cfg=cfg) + fphi_loc

    if return_sorted:
        return particlesz.pos, particlesz.mass, isortz, fphi
    else:
        fphi_unsorted = jnp.zeros_like(fphi).at[isortz].set(fphi)
        return fphi_unsorted
new_fmm_fphi.jit = jax.jit(new_fmm_fphi, static_argnames=("cfg", "return_sorted"))

def get_force_and_potential(pos, mass, cfg : config.Config, separately=False):
    if cfg.fmm is None: # Use direct summation
        xm = jnp.concatenate([pos, mass[:,None]], axis=-1)
        fphi = cj.forces.force_and_potential(xm, softening=cfg.softening, kahan=True) * cfg.G()
    else:
        fphi = new_fmm_fphi(pos, mass, cfg=cfg) * cfg.G()

    if separately:
        return fphi[:,0:3], fphi[:,3]
    else:
        return fphi
get_force_and_potential.jit = jax.jit(get_force_and_potential, static_argnames=("cfg", "separately"))

# ------------------------------------------------------------------------------------------------ #
#                                      Tree Building Functions                                     #
# ------------------------------------------------------------------------------------------------ #

def coarsen_plane(fine: TreePlane, cfg : Config) -> TreePlane:
    """Gets the next coarser tree plane from a finer one"""
    max_size = int(fine.max_node_size * cfg.fmm.coarse_fac)
    
    res = fmdj.ztree.summarize_leaves(
        fine.cent, fine.npart, max_size=max_size, num_part=fine.tot_npart,
        ref_fac=cfg.fmm.coarse_fac, alloc_fac_nodes=cfg.fmm.alloc_fac_nodes
    )

    coarse = TreePlane(*res, max_node_size = max_size, tot_npart = fine.tot_npart, size_children = fine.size())

    if fine.mp is not None:
        coarse.mp = coarsen_multipoles(fine.mp, coarse, cfg=cfg)

    return coarse
coarsen_plane.jit = jax.jit(coarsen_plane, static_argnames=['cfg'])

def build_tree_hierarchy(part: PosMass, cfg: Config) -> list[TreePlane]:
    with_multipoles = cfg.fmm.p > 0

    res = fmdj.ztree.summarize_leaves(
        part.pos, max_size=cfg.fmm.max_leaf_size, num_part=part.pos.shape[0],
        alloc_fac_nodes=cfg.fmm.alloc_fac_nodes
    )
    leaves = TreePlane(*res, max_node_size=cfg.fmm.max_leaf_size, tot_npart=part.pos.shape[0], size_children=len(part.pos))
    if with_multipoles:
        leaves.mp = multipoles_from_particles(leaves, part, cfg=cfg)

    tree_levels : list[TreePlane] = [leaves]

    new_level = leaves
    while len(new_level.lvl) > cfg.fmm.stop_coarsen:
        new_level = coarsen_plane.jit(tree_levels[-1], cfg)
        tree_levels.append(new_level)
    return tree_levels
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg'])

def evaluate_interaction_hierarchy(th, cfg):
    ilist, loc, last_plane = None, None, None
    for i in reversed(range(0, len(th))):
        loc, ilist = evaluate_plane_interactions(th[i], last_plane, ilist, loc, cfg=cfg)
        last_plane = th[i]
    return loc, ilist
evaluate_interaction_hierarchy.jit = jax.jit(evaluate_interaction_hierarchy, static_argnames=['cfg'])