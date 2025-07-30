import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental import checkify
import jax.numpy as jnp
from dataclasses import dataclass, fields

if jax.__version__ <= "0.4.2":
    raise ImportError("This code requires JAX version 0.4.2 or higher. " \
                      "However, we could define a custom decorator for this case")

@partial(jax.tree_util.register_dataclass, 
         data_fields=["lbound", "rbound", "lchild", "rchild", "level_binary"], 
         meta_fields=[])
@dataclass
class BinaryTree:
    # Jax arrays
    lbound: jnp.ndarray = None
    rbound: jnp.ndarray = None
    lchild: jnp.ndarray = None
    rchild: jnp.ndarray = None

    level_binary: jnp.ndarray = None

@partial(jax.tree_util.register_dataclass, 
         data_fields=["lchild", "rchild", "is_valid", "level", "xnode", 
                      "xleaf", "mp", "leaf_particle_bounds"], 
         meta_fields=["p"])
@dataclass
class Octree:
    # Jax arrays
    lchild: jnp.ndarray = None
    rchild: jnp.ndarray = None
    is_valid: jnp.ndarray = None

    level: jnp.ndarray = None

    xnode: jnp.ndarray = None
    xleaf: jnp.ndarray = None

    mp: jnp.ndarray = None
    leaf_particle_bounds: jnp.ndarray = None

    p: int = 0

def pos_to_icoord(pos, L=1., bits=30):
    """Converts position in (-L/2, L/2) to integer grid coordinates"""
    valid = jnp.all((pos[...,0:3] >= -L/2.) & (pos[...,0:3] < L), axis=-1)
    # Adding 2**30 corresponds to L/2 shift:
    # We do this in integers (and not in floats) to not lose precision at x=0
    ipos = jnp.floor(pos[...,0:3] / L * 2.0**bits).astype(jnp.int32) + 2**(bits-1)  
    
    return ipos, valid

def expand_10b(v):
    """
    Expands 10-bit integers into 30-bit by inserting two 0 bits between 
    each input bit.

    https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-3d-coordinates-for-32-bit
    """
    v = v & 0x3ff
    v = (v | v << 16) & 0x30000ff
    v = (v | v << 8) & 0x300f00f
    v = (v | v << 4) & 0x30c30c3
    v = (v | v << 2) & 0x9249249

    return v

def morton_3int32_to_3int32(ipos):
    """
    converts integer coordinates (range 0...2**30-1) to 3 bit-interleaved morton keys
    the primary key is the last one (to align with jax.lexsort)
    """
    ix, iy, iz = ipos[...,0], ipos[...,1], ipos[...,2]
    i1 = (expand_10b(ix      ) << 2) | (expand_10b(iy      ) << 1) | expand_10b(iz      )
    i2 = (expand_10b(ix >> 10) << 2) | (expand_10b(iy >> 10) << 1) | expand_10b(iz >> 10)
    i3 = (expand_10b(ix >> 20) << 2) | (expand_10b(iy >> 20) << 1) | expand_10b(iz >> 20)

    return jnp.stack([i1, i2, i3], axis=-1)

def organize_particles(pos, return_sorted=True):
    ipos, valid = pos_to_icoord(pos)
    morton = morton_3int32_to_3int32(ipos)
    isort = jnp.lexsort(morton.T)
    if return_sorted:
        return morton[isort], pos[isort], isort
    else:
        return morton, isort