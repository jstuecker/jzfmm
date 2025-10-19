import fmdj
import custom_jax as cj
import jax
import jax.numpy as jnp

import dataclasses
from jax.tree_util import register_dataclass

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TreeLevel():
    """A tree level defines the relation between nodes and their children."""

    # Defined per node:
    ispl: jnp.ndarray # relation to children

    npart: jnp.ndarray
    lvl: jnp.ndarray
    cent: jnp.ndarray

    # Scalars (data dependent)
    nnodes: jnp.ndarray

    # metadata (data independent)
    max_size: int
    num_part: int

    def icoarse_of_fine(self, nfine) -> jnp.ndarray:
        return jnp.searchsorted(self.ispl, jnp.arange(nfine))
    
@dataclasses.dataclass(frozen=True)
class TreeConfig():
    max_leaf_size : int = 64
    alloc_fac_nodes : float = 1.0
    alloc_min : int = 128
    coarse_fac : float = 16.0
    stop_coarsen : int = 512

def coarsen_level(fine: TreeLevel, cfg : TreeConfig) -> TreeLevel:
    """Coarsen a tree level to form the next coarser level."""
    max_size = int(fine.max_size * cfg.coarse_fac)
    
    res = cj.tree.summarize_leaves(
        fine.cent, fine.npart, max_size=max_size, num_part=fine.num_part,
        ref_fac=cfg.coarse_fac, alloc_fac_nodes=cfg.alloc_fac_nodes
    )

    coarse = TreeLevel(*res, max_size = max_size, num_part = fine.num_part)

    return coarse
coarsen_level.jit = jax.jit(coarsen_level, static_argnames=['cfg'])

def build_level_hierarchy(posz, cfg : TreeConfig) -> list[TreeLevel]:

    res = cj.tree.summarize_leaves(
        posz, max_size=cfg.max_leaf_size, num_part=posz.shape[0],
        ref_fac=cfg.coarse_fac, alloc_fac_nodes=cfg.alloc_fac_nodes
    )
    leaves = TreeLevel(*res, max_size=cfg.max_leaf_size, num_part=posz.shape[0])

    tree_levels : list[TreeLevel] = [leaves]

    new_level = leaves
    while len(new_level.lvl) > cfg.stop_coarsen:
        new_level = coarsen_level(tree_levels[-1], cfg)
        tree_levels.append(new_level)
    return tree_levels
build_level_hierarchy.jit = jax.jit(build_level_hierarchy, static_argnames=['cfg'])