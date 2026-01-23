from dataclasses import dataclass, field
from typing import List
import jax
import jax.numpy as jnp

from jztree.tools import cumsum_starting_with_zero, inverse_of_splits
from jztree.data import PosMass, Pos, InteractionList

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