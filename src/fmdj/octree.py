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

