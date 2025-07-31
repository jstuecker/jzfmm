import jax
import jax.numpy as jnp
from .octree import Octree, get_parent_binary

def save_divide(a, b):
    return jnp.where(b != 0, a / b, 0.)

def com_via_levels(octree : Octree, pos, mass):
    """Computes the mass and the center of mass of each node in the octree"""
    max_nodes = len(octree.lchild)

    # First add particles into their parent nodes
    m = jnp.zeros(max_nodes, dtype=jnp.float32
                  ).at[octree.node_of_particle].add(mass)
    mx = jnp.zeros((max_nodes, 3), dtype=jnp.float32
                   ).at[octree.node_of_particle].add(pos * mass[:,None])

    # Next propagate information up the tree
    def handle_level(i, carry):
        level_parent = -i
        m, mx = carry

        sel = (octree.level_binary[octree.parent] == level_parent) & octree.is_valid
        ipar = jnp.where(sel, octree.parent, max_nodes)
        
        m = m.at[ipar].add(m)
        mx = mx.at[ipar].add(mx)

        return m, mx
    
    m, mx = jax.lax.fori_loop(-90, 1, handle_level, (m, mx))

    return m, save_divide(mx, m[:,None])