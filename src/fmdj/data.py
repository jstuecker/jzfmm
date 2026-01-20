import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Iterator
from .tools import cumsum_starting_with_zero, inverse_of_splits, offset_sum, masked_prefix_sum

def static_field(*args, **kwargs):
    return field(*args, metadata=dict(static=True), **kwargs)

@jax.tree_util.register_dataclass
@dataclass
class LocalExpansion:
    values: jax.Array

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
class Pos: # this class is mostly defined to declare an interface that particle class should follow
    pos: jax.Array

@jax.tree_util.register_dataclass
@dataclass
class PosMass:
    pos: jax.Array  # (Nparticles, 3)
    mass: jax.Array  # (Nparticles,)

    def posm(self):
        return jnp.concatenate([self.pos, self.mass[:, None]], axis=-1)

@jax.tree_util.register_dataclass
@dataclass
class Particles(PosMass):
    vel : jax.Array
    loc : LocalExpansion | None = None

    cpos : jax.Array | None = None
    cvel : jax.Array | None = None

    def apos(self):
        return self.pos if self.cpos is None else self.pos + self.cpos
    def avel(self):
        return self.vel if self.cvel is None else self.vel + self.cvel

@jax.tree_util.register_dataclass
@dataclass
class PackedArray:
    data: jax.Array
    ispl: jax.Array
    fill_values: jax.Array | None = None
    # keep levels_filled as an array with shape (1,) for compatibility with shard map:
    levels_filled: jax.Array = field(default_factory=lambda: jnp.zeros((1,), dtype=jnp.int32))

    # Alternate constructors
    @classmethod
    def create_empty(cls, size, levels: int, dtype=jnp.float32, *, fill_values=None, vma=None):
        data = jnp.zeros(size, dtype=dtype)
        ispl = jnp.zeros(levels + 1, dtype=jnp.int32)
        levels_filled = jnp.zeros((1,), dtype=jnp.int32)

        if vma is not None:
            data = jax.lax.pcast(data, tuple(vma), to="varying")
            ispl = jax.lax.pcast(ispl, tuple(vma), to="varying")

        if jnp.isscalar(fill_values):
            fill_values = jnp.full(levels, fill_values, dtype=dtype)

        return cls(data, ispl, fill_values, levels_filled)
    
    @classmethod
    def from_data(cls, data, ispl, fill_values=None, levels_filled=None):
        if levels_filled is None:
            levels_filled = jnp.full((1,), len(ispl)-1, dtype=jnp.int32)
        else:
            levels_filled = jnp.asarray(levels_filled).reshape(1,)

        if jnp.isscalar(fill_values):
            fill_values = jnp.full(len(ispl)-1, fill_values, dtype=data.dtype)

        return cls(data, ispl, fill_values, levels_filled)

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
        return PackedArray(new_data, new_spl, new_fill_vals, jnp.reshape(level+1, (1,)))
    
    def append(self, values, num=None, fill_value=None):
        return self.set(self.levels_filled[0], values, num, fill_value)
    
    def size(self):
        return len(self.data)
    
    def num(self, level):
        return self.ispl[level + 1] - self.ispl[level]
    
    def nfilled(self):
        return self.ispl[-1]
    
    def nlevels(self):
        return len(self.ispl) - 1
    
    def all_equal(self, other: "PackedArray"):
        def eq_nan(x, y):
            return (x == y) | (jnp.isnan(x) & jnp.isnan(y))

        equal  = jnp.all(eq_nan(self.data, other.data))
        equal &= jnp.all(self.ispl == other.ispl)
        equal &= jnp.all(eq_nan(self.fill_values, other.fill_values))
        equal &= jnp.all(self.levels_filled == other.levels_filled)
        return equal

@jax.tree_util.register_dataclass
@dataclass
class TreePlane():
    # Defined per node:
    ispl: jax.Array # relation to children

    npart: jax.Array
    lvl: jax.Array
    geom_cent: jax.Array

    # Scalars (data dependent)
    nnodes: jax.Array

    around_com: bool = static_field()

    # Optional data:
    mass_cent: PosMass | None = None      # Optionally needed

    def size(self) -> int: # needed
        return self.lvl.shape[0]
    def center(self) -> jax.Array:
        if self.around_com:
            assert self.mass_cent is not None, "Mass center not available"
            return self.mass_cent.pos
        else:
            return self.geom_cent
    def node_extent(self, diag2=False) -> jax.Array: # only jax
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
class TreeHierarchy():
    # Packed Arrays:
    ispl_n2n: PackedArray
    ispl_n2l: PackedArray

    # tree plane data:
    lvl: PackedArray
    geom_cent: PackedArray
    mass: PackedArray | None = None
    mass_cent: PackedArray | None = None

    plane_sizes: List[int] = static_field(default_factory=list)

    def npart(self, level: int, size=None) -> jax.Array:
        if size is None:
            size = self.plane_sizes[level]
        ispl_n2p = self.ispl_n2n.get(0)[self.ispl_n2l.get(level, size+1)]
        return ispl_n2p[1:] - ispl_n2p[:-1]

    def center(self) -> PackedArray:
        if self.mass_cent is not None:
            return self.mass_cent
        else:
            return self.geom_cent
        
    def get_tree_plane(self, level: int, size=None) -> TreePlane:
        if size is None:
            size = self.plane_sizes[level]

        ispl_n2p = self.ispl_n2n.get(0)[self.ispl_n2l.get(level, size)]
        if self.mass_cent is not None:
            mass_cent = PosMass(self.mass_cent.get(level, size), self.mass.get(level, size))
        else:
            mass_cent = None
        return TreePlane(
            ispl = self.ispl_n2n.get(level, size+1),
            npart = ispl_n2p[1:] - ispl_n2p[:-1],
            lvl = self.lvl.get(level, size),
            geom_cent = self.geom_cent.get(level, size),
            nnodes = self.lvl.num(level),
            around_com = self.mass_cent is not None,
            mass_cent = mass_cent
        )
    
    def num_planes(self) -> int:
        return len(self.ispl_n2l.ispl) - 1
    
    def planes(self) -> Iterator[TreePlane]:
        for level in range(self.num_planes()):
            yield self.get_tree_plane(level)

    def num(self, level) -> int:
        return self.lvl.num(level)

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

    ispl: List[jax.Array]

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
    ispl: jax.Array
    iother: jax.Array

    # Multi-GPU specific: origin ids and device offets
    ids: jax.Array | None = None
    dev_spl: jax.Array | None = None

    def get_interactions(self, get_valid=False):
        """Returns (i0, i1, valid) indicating two interaction nodes and validity"""
        iint = jnp.arange(self.size(), dtype=self.dtype())
        i0 = inverse_of_splits(self.ispl, self.size())
        i1 = self.iother#[iint]
        if get_valid:
            valid = iint < self.ispl[-1]
            return i0, i1, valid
        else:
            return i0, i1
    
    def filter(self, mask: jax.Array, size: int | None = None) -> 'InteractionList':
        """Returns a filtered interaction list according to the boolean mask"""
        if size is None:
            size = mask.size
        ioff = cumsum_starting_with_zero(mask)
        iupdate = jnp.where(mask, ioff, size)
        iother_new = jnp.zeros(size, dtype=self.iother.dtype).at[iupdate].set(self.iother)
        ispl_new = ioff[self.ispl]

        return InteractionList(ispl=ispl_new, iother=iother_new)
    
    def nfilled(self):
        return self.ispl[-1]

    def size(self):
        return self.iother.size
    
    def dtype(self):
        return self.iother.dtype

def set_range(arr : jax.Array, values, start, end):
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