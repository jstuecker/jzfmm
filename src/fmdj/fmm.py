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
from fmdj.ztree import build_tree_hierarchy

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

def evaluate_interaction_hierarchy(th, cfg):
    ilist, loc, last_plane = None, None, None
    for i in reversed(range(0, len(th))):
        loc, ilist = evaluate_plane_interactions(th[i], last_plane, ilist, loc, cfg=cfg)
        last_plane = th[i]
    return loc, ilist
evaluate_interaction_hierarchy.jit = jax.jit(evaluate_interaction_hierarchy, static_argnames=['cfg'])