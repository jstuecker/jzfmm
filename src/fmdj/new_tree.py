import custom_jax as cj
import jax
import jax.numpy as jnp
from fmdj.multipoles import x_moment, shift_multipoles, shift_multipoles
from dataclasses import dataclass, field
from .config import Config, TreeConfig

from .variants import vm, make_dispatcher, V, VariantConfig

def static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)

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
    def node_extent(self) -> jnp.ndarray:
        return jnp.ldexp(1., self.lvl)

@jax.tree_util.register_dataclass
@dataclass
class Particles:
    pos: jnp.ndarray  # (Nparticles, 3)
    mass: jnp.ndarray  # (Nparticles,)

    def posm(self):
        return jnp.concatenate([self.pos, self.mass[:, None]], axis=-1)

def coarsen_plane(fine: TreePlane, cfg : Config) -> TreePlane:
    """Gets the next coarser tree plane from a finer one"""
    cfg_tree: TreeConfig = cfg.tree

    max_size = int(fine.max_node_size * cfg_tree.coarse_fac)
    
    res = cj.tree.summarize_leaves(
        fine.cent, fine.npart, max_size=max_size, num_part=fine.tot_npart,
        ref_fac=cfg_tree.coarse_fac, alloc_fac_nodes=cfg_tree.alloc_fac_nodes
    )

    coarse = TreePlane(*res, max_node_size = max_size, tot_npart = fine.tot_npart, size_fine = len(fine.lvl))

    if fine.mp is not None:
        coarse.mp = coarsen_multipoles(fine.mp, coarse, cfg=cfg)

    return coarse
coarsen_plane.jit = jax.jit(coarsen_plane, static_argnames=['cfg'])

def build_tree_hierarchy(part: Particles, cfg: Config) -> list[TreePlane]:
    cfg_tree: TreeConfig = cfg.tree
    with_multipoles = cfg_tree.p > 0

    res = cj.tree.summarize_leaves(
        part.pos, max_size=cfg_tree.max_leaf_size, num_part=part.pos.shape[0],
        alloc_fac_nodes=cfg_tree.alloc_fac_nodes
    )
    leaves = TreePlane(*res, max_node_size=cfg_tree.max_leaf_size, tot_npart=part.pos.shape[0], size_fine=len(part.pos))
    if with_multipoles:
        leaves.mp = multipoles_from_particles(leaves, part, cfg=cfg)

    tree_levels : list[TreePlane] = [leaves]

    new_level = leaves
    while len(new_level.lvl) > cfg_tree.stop_coarsen:
        new_level = coarsen_plane.jit(tree_levels[-1], cfg)
        tree_levels.append(new_level)
    return tree_levels
build_tree_hierarchy.jit = jax.jit(build_tree_hierarchy, static_argnames=['cfg'])

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

def _multipoles_from_particles_base(tp: TreePlane, part: Particles, *, cfg: Config) -> Multipoles:
    cfg_tree: TreeConfig = cfg.tree

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
    
    if cfg_tree.multipoles_around_com:
        xcent = jnp.stack([mxnode[d]/mnode for d in range(3)], axis=-1) + tp.geom_center()
        mp.extend([jnp.zeros_like(mnode, dtype=dtype)]*3)
    else:
        xcent = tp.geom_center()
        mp.extend(mxnode)
    dx = part.pos - xcent[parent]

    # Compute multipole moments
    for i, nvec in iterate_multi_indices(cfg_tree.p, istart=4):
        mp.append(jax.ops.segment_sum(x_moment(dx, nvec) * part.mass, **kwargs))

    return Multipoles(
        xcent=xcent,
        values=jnp.stack(mp, axis=-1),
        p=cfg_tree.p,
        around_com=cfg_tree.multipoles_around_com
    )

def _coarsen_multipoles_base(mp: Multipoles, tp: TreePlane, *, cfg: Config) -> Multipoles:
    """Determines the multipoles at the next coarser tree plane"""
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

# ------------------------------------------------------------------------------------------------ #
#                                         Interaction Lists                                        #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass
class InteractionList:
    """Node i0 will interact with all indices iother[ispl[i0]:ispl[i0+1]]"""
    ispl: jnp.ndarray
    iother: jnp.ndarray

    nfilled : jnp.ndarray  # Total number of filled interactions

    def get_interaction_range(self, a, b):
        """Returns (i0, i1, valid) indicating two interaction nodes and validity"""
        iint = jnp.arange(b - a, dtype=self.iother.dtype) + a
        i0 = jnp.searchsorted(self.ispl, iint, side='right') - 1
        i1 = self.iother[iint]
        valid = iint < self.nfilled
        return i0, i1, valid
    
    def size(self):
        return self.iother.size


def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def masked_prefix_sum(mask):
    off, n = offset_sum(mask)
    off_masked = jnp.where(mask, off, len(mask))
    return off_masked, n

def dense_interaction_list(size: int, nnodes: jnp.ndarray = None) -> InteractionList:
    """A dense interaction list where all nodes interact with all other nodes.

    size: size of the node array that will use the interaction list. (Required at compile time)
    nnodes: actual number of filled nodes (Can be dynamic, used to invalidating unused nodes)
    """

    if nnodes is None: # size = nnodes will only work outside of jit
        nnodes = jnp.array(size, dtype=jnp.int32)  
    dtype = nnodes.dtype

    # We need to work around JAX's lack of dynamic array sizes
    i1, i2 = jnp.indices((size, size), dtype=dtype)
    
    valid = (i1 < nnodes) & (i2 < nnodes)

    ioff, nfilled = masked_prefix_sum(valid.flatten())

    ilist = jnp.zeros(i1.size, dtype=i1.dtype).at[ioff].set(i2.flatten())

    ispl = jnp.arange(0, size, dtype=i1.dtype) * nnodes
    ispl = jnp.where(ispl < nfilled, ispl, nfilled)
    
    return InteractionList(ispl=ispl, iother=ilist, nfilled=nfilled)
dense_interaction_list.jit = jax.jit(dense_interaction_list, static_argnames=['size'])

# ------------------------------------------------------------------------------------------------ #
#                                             Tree Walk                                            #
# ------------------------------------------------------------------------------------------------ #

def norm2(dx: jnp.ndarray) -> jnp.ndarray:
    return dx[...,0]**2 + dx[...,1]**2 + dx[...,2]**2

def opening_criterion_bnh(plane: TreePlane, i0: jnp.ndarray, i1: jnp.ndarray, cfg: Config):
    """Barnes & Hut Opening Criterion."""
    theta = cfg.opening.opening_angle

    r2 = norm2(plane.cent[i1] - plane.cent[i0])

    L0, L1 = plane.node_extent()[i0], plane.node_extent()[i1]

    need_open = (L0 + L1)**2 > theta**2 * r2

    print(jnp.nanmin((L0 + L1)**2 / r2), theta**2)
    print(jnp.nanmean(need_open))

    return need_open

# ------------------------------------------------------------------------------------------------ #
#                                       Register Dispatchers                                       #
# ------------------------------------------------------------------------------------------------ #

multipoles_from_particles = make_dispatcher(vm[V.multipoles_from_particles], _multipoles_from_particles_base, add_jit=True)
coarsen_multipoles = make_dispatcher(vm[V.coarsen_multipoles], _coarsen_multipoles_base, add_jit=True)


