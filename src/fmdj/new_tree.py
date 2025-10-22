import custom_jax as cj
import jax
import jax.numpy as jnp
from fmdj.multipoles import x_moment, shift_multipoles, shift_local_to_local, ilist_node_to_node
from dataclasses import dataclass, field
from .config import Config, TreeConfig
from fmdj.tools import conditional_callback
from typing import Tuple
import fmdj

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

    size_children : int = static_field()

    # Optional data:
    mp: Multipoles = None

    def icoarse_of_fine(self) -> jnp.ndarray:
        return jnp.searchsorted(self.ispl, jnp.arange(self.size_children), side="right") - 1
    def geom_center(self) -> jnp.ndarray:
        return self.cent
    def size(self) -> int:
        return self.lvl.shape[0]
    def node_extent(self, diag2=False) -> jnp.ndarray:
        # return jnp.ldexp(1., self.lvl)
        olvl, omod = self.lvl//3, self.lvl % 3

        dx = jnp.ldexp(1., olvl)
        dy = jnp.ldexp(1., olvl + (omod >= 2).astype(jnp.int32))
        dz = jnp.ldexp(1., olvl + (omod >= 1).astype(jnp.int32))

        if diag2:
            return dx*dx + dy*dy + dz*dz
        else:
            return jnp.stack((dx, dy, dz), axis=-1)
    
def lvl_to_ext(level_binary):
    olvl, omod = level_binary//3, level_binary % 3
    levels_3d = jnp.stack((olvl, olvl + (omod >= 2).astype(jnp.int32), olvl + (omod >= 1).astype(jnp.int32)),axis=-1)
    return 2.**levels_3d

    def get_child_indices(self, nchild : jnp.ndarray = None):
        """Returns (igroup, idx, valid) indicating indices of group, child and validity"""
        if nchild is None:
            nchild = self.size_children
        isub = jnp.arange(self.size_children, dtype=self.ispl.dtype)
        igroup = jnp.searchsorted(self.ispl, isub, side='right') - 1
        valid = (igroup < self.nnodes) & (isub < nchild)
        return igroup, isub, valid

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

    coarse = TreePlane(*res, max_node_size = max_size, tot_npart = fine.tot_npart, size_children = fine.size())

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
    leaves = TreePlane(*res, max_node_size=cfg_tree.max_leaf_size, tot_npart=part.pos.shape[0], size_children=len(part.pos))
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

from typing import List

def find_group(ispl, index):
    return jnp.searchsorted(ispl, index, side='right') - 1

def inverse_of_splits(ispl, size):
    """given [0, 4, 7] returns [0,0,0,0,1,1,1] for size=7"""
    mask = jnp.zeros(size, dtype=jnp.int32).at[ispl].set(1)
    return jnp.cumsum(mask) - 1

@jax.tree_util.register_dataclass
@dataclass
class SegmentedNDArray():
    """This class provides index mapping strategies to define ndarrays with locally varying shapes
    
    We define splitting points so that the elements that have first index i0 are located
    between spl[i0] and spl[i0+1] on the next finer level.

    For example, consdier an ndarray x with shape (4,3), we could represent it through splits
    spl[0] = [0, 3, 6, 9, 12]. E.g. to get x[a,b] you could use x.flat[multi_to_flat(a,b)] to 
    index it. However, here we can also create adaptively shaped arrays.

    To also support higher dimensions than 2, it is possible to define a hierarchy of splits. For
    example a (4, 3, 2) array would be represented through
    spl[0] = [0, 3, 6, 9, 12]
    spl[1] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

    For nd-arrays the shape of each split corresponds to the product of lower levels shapes + 1
    and the largest element in the split is the product of the lower levels + the new one
    """

    ispl: List[jnp.ndarray]

    def global_ispl(self, axis=0):
        if axis == len(self.ispl) - 1:
            return self.ispl[axis]
        else:
            ispl = self.global_ispl(axis=axis+1)
            return ispl[self.ispl[axis]]

    def n(self, idx=None, axis=0):
        if idx is None:
            return self.ispl[axis][1:] - self.ispl[axis][:-1]
        else:
            # Note: if idx is out ouf bounds this will return 0
            return self.ispl[axis][idx+1] - self.ispl[axis][idx]
    
    def multi_indices(self, size, get_valid=False):
        return self.flat_to_multi(jnp.arange(size, dtype=self.ispl[0].dtype), get_valid=get_valid)

    def flat_to_multi(self, iflat, get_valid=False):
        imult = []
        valid = True
        for ispl in self.ispl[::-1]:
            valid = valid & (iflat < ispl[-1])
            igroup = inverse_of_splits(ispl, iflat.shape[0])[iflat]
            imult.append(iflat - ispl[igroup])
            iflat = igroup
        imult.append(iflat)
        if get_valid:
            return imult[::-1], valid
        else:
            return imult[::-1]
    
    def multi_to_flat(self, imult, get_valid=False):
        assert len(imult) == len(self.ispl) + 1
        valid = jnp.ones_like(imult[0], dtype=bool)
        iflat = jnp.zeros_like(imult[0])
        for i, ispl in enumerate(self.ispl):
            iflat = ispl[iflat + imult[i]]
            valid = valid & (iflat < ispl[-1])
        if get_valid:
            return iflat + imult[-1], valid
        else:
            return iflat + imult[-1]
    
    def expand(self, n):
        """Segmentation that is given by replicating each element n[i] times"""
        ispl_new = cumsum_starting_with_zero(n)
        return SegmentedNDArray(ispl=[*self.ispl, ispl_new])

@jax.tree_util.register_dataclass
@dataclass
class InteractionList:
    """Node i0 will interact with all indices iother[ispl[i0]:ispl[i0+1]]"""
    ispl: jnp.ndarray
    iother: jnp.ndarray

    nfilled : jnp.ndarray  # Total number of filled interactions

    def get_interactions(self, get_valid=False):
        """Returns (i0, i1, valid) indicating two interaction nodes and validity"""
        iint = jnp.arange(self.size(), dtype=self.dtype())
        i0 = jnp.searchsorted(self.ispl, iint, side='right') - 1
        i1 = self.iother[iint]
        valid = iint < self.nfilled
        if get_valid:
            return i0, i1, valid
        else:
            return i0, i1
    
    def filter(self, mask: jnp.ndarray, size: int | None = None) -> 'InteractionList':
        """Returns a filtered interaction list according to the boolean mask"""
        if size is None:
            size = self.iother.size
        ioff, nfilled = offset_sum(mask)
        iother_new = jnp.zeros(size, dtype=self.iother.dtype).at[ioff].set(self.iother)
        ispl_new = ioff[self.ispl]

        return InteractionList(ispl=ispl_new, iother=iother_new, nfilled=nfilled)
    
    def size(self):
        return self.iother.size
    
    def dtype(self):
        return self.iother.dtype

def expand_interactions(
        ilist: InteractionList, 
        ispl: jnp.ndarray, 
        size_children: int, 
        size_new_ilist: int) -> InteractionList:
    """Expands the interaction list to the children"""
    # This works by adding two (variable size) extra dimensions to the interaction list.
    # Since the indexing logic of segmented arrays is rather complicated, we use a 
    # helper class SegmentedNDArray to handle the indexing.

    # Helper, Node->interaction
    seg_ilist = SegmentedNDArray(ispl=[ilist.ispl])

    # Node->Child segments
    seg_nodes = SegmentedNDArray(ispl=[ispl])

    # Expand to Node->Child->Nodeinteraction
    (in0, ic0), valid = seg_nodes.multi_indices(size_children, get_valid=True)
    node_exp = seg_nodes.expand(seg_ilist.n(in0) * valid)

    # Expand to Node->Child->Nodeinteraction->Otherchild
    # Note: this one is only needed temporarily and could in principle use a smaller size
    (in0, ic0, iint), valid = node_exp.multi_indices(size_new_ilist, get_valid=True)
    i1 = ilist.iother[seg_ilist.multi_to_flat((in0, iint))]
    node_cc = node_exp.expand(seg_nodes.n(i1) * valid)

    # Get interaction list
    (in0, ic0, iint, ic1), valid = node_cc.multi_indices(size_new_ilist, get_valid=True)
    inode1 = ilist.iother[seg_ilist.multi_to_flat((in0, iint))]
    iother_new = jnp.where(valid, seg_nodes.multi_to_flat((inode1, ic1)), size_new_ilist)

    # Discard last dimension
    ispln = node_cc.global_ispl(1)

    # Check that the sizes are big enough
    def size_error(nfilled, size):
        raise ValueError(f"Expanded interaction list ({nfilled}) does not fit into buffer ({size})")

    ispln = ispln + conditional_callback(ispln[-1] >= size_new_ilist, size_error, ispln[-1], size_new_ilist)
    
    return InteractionList(ispl = ispln, iother = iother_new, nfilled = ispln[-1])
expand_interactions.jit = jax.jit(expand_interactions, static_argnames=['size_children', 'size_new_ilist'])


def cumsum_starting_with_zero(x):
    return jnp.concatenate((jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), jnp.cumsum(x, axis=0)))

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

    r2 = norm2(plane.mp.center()[i1] - plane.mp.center()[i0])

    # L0, L1 = plane.node_extent()[i0], plane.node_extent()[i1]
    L0, L1 = plane.node_extent()[i0], plane.node_extent()[i1]

    need_open = norm2(L0 + L1) > theta**2 * r2
    # To avoid dealing with overflow issues, we open very large nodes explicitly:
    need_open = need_open | (plane.lvl[i1] >= 150) | (plane.lvl[i0] >= 150)

    return need_open

def evaluate_plane_interactions(plane: TreePlane, 
                                plane_lr: TreePlane | None = None,
                                ilist_lr: InteractionList | None = None,
                                loc_lr: jnp.ndarray | None = None,
                                cfg: Config = None
                                ) -> Tuple[jnp.ndarray, InteractionList]:
    """Evaluate all interactions for a given tree plane."""
    if plane_lr is None or ilist_lr is None: # Root level
        ilist = dense_interaction_list(plane.size(), nnodes=plane.nnodes)
    else:
        # Add all child-child pairs for each coarser interaction
        ilist_size_new = cfg.tree.ilist_alloc_fac * plane.size()
        ilist = expand_interactions(ilist_lr, plane_lr.ispl, plane.size(), ilist_size_new)

    # Interaction indices
    i0, i1, valid = ilist.get_interactions(get_valid=True)

    need_open = opening_criterion_bnh(plane, i0, i1, cfg=cfg)

    ilist_open = ilist.filter(valid & need_open)
    ilist_eval = ilist.filter(valid & ~need_open)
    
    # Evaluate multipole interactions
    interactions = jnp.stack(ilist_eval.get_interactions(get_valid=False), axis=-1)
    irange = jnp.stack([0, ilist_eval.nfilled])

    loc = ilist_node_to_node(plane.mp.center(), plane.mp.values, interactions, irange, cfg=cfg)
    if loc_lr is not None:
        x0 = plane_lr.mp.center()[plane_lr.icoarse_of_fine()]
        loc = loc + shift_local_to_local(loc, plane.mp.center() - x0)

    # Some logging
    open_frac = ilist_open.nfilled / ilist.nfilled
    fmdj.log("Interactions opened {}/{} ({:.1%}) sizefac: {:.1f} ({:.1%} of allocation)", 
             ilist_open.nfilled, ilist.nfilled, open_frac, ilist.nfilled / plane.size(), 
             ilist.nfilled / ilist.size(), level=2, cfg=cfg)
    
    return loc, ilist_open
evaluate_plane_interactions.jit = jax.jit(evaluate_plane_interactions, static_argnames=['cfg'])

# ------------------------------------------------------------------------------------------------ #
#                                       Register Dispatchers                                       #
# ------------------------------------------------------------------------------------------------ #

multipoles_from_particles = make_dispatcher(vm[V.multipoles_from_particles], _multipoles_from_particles_base, add_jit=True)
coarsen_multipoles = make_dispatcher(vm[V.coarsen_multipoles], _coarsen_multipoles_base, add_jit=True)


