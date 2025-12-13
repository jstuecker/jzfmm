import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List
from .tools import cumsum_starting_with_zero, inverse_of_splits, offset_sum, masked_prefix_sum

def static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)

@jax.tree_util.register_dataclass
@dataclass
class LocalExpansion:
    values: jnp.ndarray

    def fphi(self):
        return jnp.concatenate([self.force(), self.potential()[...,None]], axis=-1)
    def potential(self):
        return self.values[:, 0]
    def force(self):
        assert self.values.shape[1] >= 4, "Force components not available"
        return -self.values[:, 1:4]
    def tide(self):
        assert self.values.shape[1] >= 10, "Tidal components not available"
        return -self.values[:, 4:10]

@jax.tree_util.register_dataclass
@dataclass
class PosMass:
    pos: jnp.ndarray  # (Nparticles, 3)
    mass: jnp.ndarray  # (Nparticles,)

    def posm(self):
        return jnp.concatenate([self.pos, self.mass[:, None]], axis=-1)

@jax.tree_util.register_dataclass
@dataclass
class Particles(PosMass):
    vel : jnp.ndarray
    loc : LocalExpansion | None = None

    cpos : jnp.ndarray | None = None
    cvel : jnp.ndarray | None = None

    def apos(self):
        return self.pos if self.cpos is None else self.pos + self.cpos
    def avel(self):
        return self.vel if self.cvel is None else self.vel + self.cvel

@jax.tree_util.register_dataclass
@dataclass
class PackedArray:
    data: jnp.ndarray
    ispl: jnp.ndarray
    fill_values: jnp.ndarray | None = None

    def __init__(self, data, ispl=None, levels=None, fill_values=None):
        assert (ispl is not None) or (levels is not None), "Either ispl or num_arr must be provided"

        self.data = data
        if ispl is not None:
            self.ispl = ispl
            levels = len(ispl) - 1
        elif levels is not None:
            self.ispl = jnp.zeros(levels + 1, dtype=jnp.int32)
        if fill_values is None:
            if data.dtype == jnp.float32 or data.dtype == jnp.float64:
                fill_values = jnp.nan
            else:
                fill_values = 0
        if jnp.isscalar(fill_values):
            self.fill_values = jnp.full(levels, fill_values, dtype=self.data.dtype)
        else:
            assert len(fill_values) == levels
            self.fill_values = fill_values
    
    def get(self, level, size=None, fill_value=None):
        if size is None:
            size = self.size()
        indices = jnp.arange(size) + self.ispl[level]
        valid = indices < self.ispl[level + 1]
        valid = valid.reshape((-1,) + (1,) * (self.data.ndim - 1))
        if fill_value is None:
            fill_value = self.fill_values[level]
        return jnp.where(valid, self.data[indices], fill_value)
    
    def set(self, level, values, num=None, fill_value=None):
        if num is None:
            num = values.shape[0]
        new_spl = jnp.where(jnp.arange(len(self.ispl)) <= level, self.ispl, self.ispl[level] + num)
        new_data = set_range(self.data, values, self.ispl[level], self.ispl[level] + num)
        if fill_value is not None:
            new_fill_vals = self.fill_values.at[level].set(fill_value)
        else:
            new_fill_vals = self.fill_values
        return PackedArray(new_data, ispl=new_spl, fill_values=new_fill_vals)
    
    def size(self):
        return self.data.size
    
    def num(self, level):
        return self.ispl[level + 1] - self.ispl[level]
    
    def nfilled(self):
        return self.ispl[-1]

@jax.tree_util.register_dataclass
@dataclass
class TreePlane():
    # Defined per node:
    ispl: jnp.ndarray # relation to children

    npart: jnp.ndarray
    lvl: jnp.ndarray
    geom_cent: jnp.ndarray

    # Scalars (data dependent)
    nnodes: jnp.ndarray

    around_com: bool = static_field()

    # Optional data:
    mass_cent: PosMass | None = None      # Optionally needed

    def size(self) -> int: # needed
        return self.lvl.shape[0]
    def center(self) -> jnp.ndarray:
        if self.around_com:
            assert self.mass_cent is not None, "Mass center not available"
            return self.mass_cent.pos
        else:
            return self.geom_cent
    def node_extent(self, diag2=False) -> jnp.ndarray: # only jax
        # return jnp.ldexp(1., self.lvl)
        olvl, omod = self.lvl//3, self.lvl % 3

        dx = jnp.ldexp(1., olvl)
        dy = jnp.ldexp(1., olvl + (omod >= 2).astype(jnp.int32))
        dz = jnp.ldexp(1., olvl + (omod >= 1).astype(jnp.int32))

        if diag2:
            return dx*dx + dy*dy + dz*dz
        else:
            return jnp.stack((dx, dy, dz), axis=-1)

@jax.tree_util.register_dataclass
@dataclass
class NewTreeHierarchy():
    # Packed Arrays:
    ispl_n2n: PackedArray
    ispl_n2l: PackedArray

    # tree plane data:
    lvl: PackedArray
    geom_cent: PackedArray
    mass_cent: PackedArray | None = None

    def center(self) -> PackedArray:
        if self.mass_cent is not None:
            return self.mass_cent
        else:
            return self.geom_cent

@jax.tree_util.register_dataclass
@dataclass
class TreeHierarchy():
    # Particles
    particles: PosMass

    # leaf specific
    leaf_ispl: jnp.ndarray
    leaf_mass_cent: PosMass

    # Node specific data
    lbound: jnp.ndarray
    rbound: jnp.ndarray
    node_npart: jnp.ndarray

    # Packed Arrays:
    p_ispl_l: PackedArray | None = None
    p_ispl_p: PackedArray | None = None
    p_mass: PackedArray | None = None
    p_mass_cent: PackedArray | None = None
    p_geom_cent: PackedArray | None = None

    def get_plane_relation(self, nsize_fine, nsize_coarse, size_fine: int, size: int) -> TreePlane:
        nnodes_fine = jnp.sum(self.node_npart > nsize_fine) - 1

        # Determine the splitting point towards the finer level
        inodes_fine = jnp.where(self.node_npart > nsize_fine, size=size_fine, fill_value=size_fine)[0]
        np_fine = self.node_npart.at[inodes_fine].get(fill_value=0)
        ispl = jnp.where(np_fine > nsize_coarse, size=size, fill_value=nnodes_fine)[0]

        return ispl
    
    def tree_plane(self, nsize_fine, nsize_coarse, size_fine: int, size: int) -> TreePlane:
        ispl = self.get_plane_relation(nsize_fine, nsize_coarse, size_fine, size)

        nleaves = jnp.argmax(self.leaf_ispl)

        nnodes = jnp.sum(self.node_npart > nsize_coarse) - 1
        ispl_l = jnp.where(self.node_npart > nsize_coarse, size=size, fill_value=nleaves)[0]

        ispl_p = self.leaf_ispl[ispl_l]
        
        from .ztree import get_node_geometry, center_of_mass
        lvl, cent, ext = get_node_geometry(self.particles.pos, ispl_p[:-1], ispl_p[1:], nnodes)

        mass_cent = center_of_mass(ispl_l, self.leaf_mass_cent)

        return TreePlane(
            ispl = ispl,
            npart = ispl_p[1:] - ispl_p[:-1], 
            lvl = lvl,
            geom_cent = cent,
            nnodes = nnodes,
            around_com = False,
            mass_cent = mass_cent
        )
    
    def leaf_plane(self) -> TreePlane:
        from .ztree import get_node_geometry

        nleaves = jnp.argmax(self.leaf_ispl)
        lvl, cent, ext = get_node_geometry(
            self.particles.pos, self.leaf_ispl[:-1], self.leaf_ispl[1:], nleaves
        )

        return TreePlane(
            ispl = self.leaf_ispl,
            npart = self.leaf_ispl[1:] - self.leaf_ispl[:-1],
            lvl = lvl,
            geom_cent = cent,
            nnodes = nleaves,
            around_com = False,
            mass_cent = self.leaf_mass_cent
        )

def find_group(ispl, index):
    return jnp.searchsorted(ispl, index, side='right') - 1

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
        i0 = inverse_of_splits(self.ispl, self.size())
        i1 = self.iother#[iint]
        if get_valid:
            valid = iint < self.nfilled
            return i0, i1, valid
        else:
            return i0, i1
    
    def filter(self, mask: jnp.ndarray, size: int | None = None) -> 'InteractionList':
        """Returns a filtered interaction list according to the boolean mask"""
        if size is None:
            size = mask.size
        ioff, nfilled = offset_sum(mask)
        iupdate = jnp.where(mask, ioff, size)
        iother_new = jnp.zeros(size, dtype=self.iother.dtype).at[iupdate].set(self.iother)
        ispl_new = ioff[self.ispl]

        return InteractionList(ispl=ispl_new, iother=iother_new, nfilled=nfilled)
    
    def size(self):
        return self.iother.size
    
    def dtype(self):
        return self.iother.dtype

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

    ispl = jnp.arange(0, size+1, dtype=i1.dtype) * nnodes
    ispl = jnp.where(ispl < nfilled, ispl, nfilled)
    
    return InteractionList(ispl=ispl, iother=ilist, nfilled=nfilled)
dense_interaction_list.jit = jax.jit(dense_interaction_list, static_argnames=['size'])

def set_range(arr : jnp.ndarray, values, start, end):
    if(len(arr) / len(values) >= 4):
        # values are much smaller than arr, do a scatter based update
        idx = jnp.arange(len(values)) + start
        idx = jnp.where(idx < end, idx, len(arr))
        return arr.at[idx].set(values, unique_indices=True)
    else:
        # Do a masked update
        idx = jnp.arange(len(arr))
        cond = (idx >= start) & (idx < end)
        cond = cond.reshape((-1,) + (1,) * (values.ndim - 1))
        return jnp.where(cond, values[idx - start], arr)