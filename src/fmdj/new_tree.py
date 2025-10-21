import custom_jax as cj
import jax
import jax.numpy as jnp
from fmdj.multipoles import x_moment, shift_multipoles, shift_multipoles
from dataclasses import dataclass, field
from typing import Callable
from functools import wraps

def static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)



@dataclass(frozen=True)
class TreeConfig():
    # important
    p : int = 2
    max_leaf_size : int = 64
    coarse_fac : float = 16.0

    # less relevant
    alloc_fac_nodes : float = 1.0
    alloc_min : int = 128
    stop_coarsen : int = 512

@jax.tree_util.register_dataclass
@dataclass
class Multipoles:
    xcent : jnp.ndarray # (3,) center of expansion
    values: jnp.ndarray # Multipoles in (M, Nnodes) layout

    p: int = static_field(default=2)

    around_com : bool = static_field(default=True)

    def center(self):
        return self.xcent
    def get(self, i):
        return self.values[:, i]

@jax.tree_util.register_dataclass
@dataclass
class TreePlane():
    # Defined per node:
    ispl: jnp.ndarray # relation to children

    npart: jnp.ndarray
    lvl: jnp.ndarray
    cent: jnp.ndarray

    # Scalars (data dependent)
    nnodes: jnp.ndarray

    # metadata (data independent)
    max_node_size: int = static_field()
    tot_npart: int = static_field()

    size_fine : int = static_field()

    # Optional data:
    mp: Multipoles = None

    def icoarse_of_fine(self) -> jnp.ndarray:
        return jnp.searchsorted(self.ispl, jnp.arange(self.size_fine), side="right") - 1
    def geom_center(self) -> jnp.ndarray:
        return self.cent
    def size(self) -> int:
        return self.lvl.shape[0]

@jax.tree_util.register_dataclass
@dataclass
class Particles:
    pos: jnp.ndarray  # (Nparticles, 3)
    mass: jnp.ndarray  # (Nparticles,)

    def posm(self):
        return jnp.concatenate([self.pos, self.mass[:, None]], axis=-1)

def coarsen_plane(fine: TreePlane, cfg : TreeConfig) -> TreePlane:
    """Gets the next coarser tree plane from a finer one"""
    max_size = int(fine.max_node_size * cfg.coarse_fac)
    
    res = cj.tree.summarize_leaves(
        fine.cent, fine.npart, max_size=max_size, num_part=fine.tot_npart,
        ref_fac=cfg.coarse_fac, alloc_fac_nodes=cfg.alloc_fac_nodes
    )

    coarse = TreePlane(*res, max_node_size = max_size, tot_npart = fine.tot_npart, size_fine = len(fine.lvl))

    if fine.mp is not None:
        coarse.mp = coarsen_multipoles(fine.mp, coarse)

    return coarse
coarsen_plane.jit = jax.jit(coarsen_plane, static_argnames=['cfg'])

def build_tree_hierarchy(part : Particles, cfg : TreeConfig, with_multipoles : bool = False) -> list[TreePlane]:
    res = cj.tree.summarize_leaves(
        part.pos, max_size=cfg.max_leaf_size, num_part=part.pos.shape[0],
        alloc_fac_nodes=cfg.alloc_fac_nodes
    )
    leaves = TreePlane(*res, max_node_size=cfg.max_leaf_size, tot_npart=part.pos.shape[0], size_fine=len(part.pos))
    if with_multipoles:
        leaves.mp = multipoles_from_particles(leaves, part, p=cfg.p)

    tree_levels : list[TreePlane] = [leaves]

    new_level = leaves
    while len(new_level.lvl) > cfg.stop_coarsen:
        new_level = coarsen_plane.jit(tree_levels[-1], cfg)
        tree_levels.append(new_level)
    return tree_levels
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg', 'with_multipoles'])

# ------------------------------------------------------------------------------------------------ #
#                                            Multipoles                                            #
# ------------------------------------------------------------------------------------------------ #

def multi_to_flat(kx, ky, kz):
    k = kx + ky + kz
    npoff = ((k+2)*(k+1)*k // 6)
    npoff += kz*(2*k + 3 - kz)//2 + ky

    return npoff

def iterate_multi_indices(p, istart=0):
    i = 0
    for n in range(p+1):
        for nz in range(n+1):
            for ny in range(n - nz +1):
                nx = n - ny - nz
                if i >= istart:
                    yield i, (nx, ny, nz)
                i += 1

def multipoles_from_particles(tp : TreePlane, part : Particles, 
                              p : int = 2, around_com : bool = True) -> Multipoles:
    dtype = part.mass.dtype

    parent = tp.icoarse_of_fine()
    kwargs = dict(
        segment_ids=parent,
        num_segments=tp.size(),
        indices_are_sorted=True
    )

    # Compute the center of mass
    mnode = jax.ops.segment_sum(part.mass, **kwargs)

    dx = part.pos - tp.geom_center()[parent]
    mxnode = [jax.ops.segment_sum(dx[...,d]*part.mass, **kwargs) for d in range(3)]

    mp = [mnode]
    
    if around_com:
        xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1) + tp.geom_center()
        mp.extend([jnp.zeros_like(mnode, dtype=dtype)]*3)
    else:
        xcent = tp.geom_center()
        mp.extend(mxnode)
    dx = part.pos - xcent[parent]

    # Compute multipole moments
    for i, nvec in iterate_multi_indices(p, istart=4):
        mp.append(jax.ops.segment_sum(x_moment(dx, nvec) * part.mass, **kwargs))

    return Multipoles(
        xcent=xcent,
        values=jnp.stack(mp, axis=-1),
        p=p,
        around_com=around_com
    )
multipoles_from_particles.jit = jax.jit(multipoles_from_particles, static_argnames=['p', 'around_com'])

def coarsen_multipoles(mp : Multipoles, tp : TreePlane) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
    dtype = mp.values.dtype

    parent = tp.icoarse_of_fine()
    kwargs = dict(
        segment_ids=parent,
        num_segments=tp.size(),
        indices_are_sorted=True
    )

    # Compute the center of mass
    mnode = jax.ops.segment_sum(mp.get(0), **kwargs)

    dx = mp.center() - tp.geom_center()[parent]
    mxnode = [jax.ops.segment_sum(dx[...,d]*mp.get(0), **kwargs) for d in range(3)]
    
    if mp.around_com:
        xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1) + tp.geom_center()
    else:
        xcent = tp.geom_center()
    
    dx = mp.center() - xcent[parent]
    
    mpshift = shift_multipoles(mp.values, dx, p=mp.p)

    mp_coarse = [jax.ops.segment_sum(mpshift[...,k], **kwargs) for k in range(mpshift.shape[-1])]

    return Multipoles(
        xcent=xcent,
        values=jnp.stack(mp_coarse, axis=-1),
        p=mp.p,
        around_com=mp.around_com
    )
coarsen_multipoles.jit = jax.jit(coarsen_multipoles)