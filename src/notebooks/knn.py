import jax
import jax.numpy as jnp

def lvl_to_ext(level_binary):
    olvl, omod = level_binary//3, level_binary % 3
    levels_3d = jnp.stack((olvl, olvl + (omod >= 2).astype(jnp.int32), olvl + (omod >= 1).astype(jnp.int32)),axis=-1)
    return 2.**levels_3d

def get_node_box(x, level_binary):
    node_size = lvl_to_ext(level_binary)
    node_cent = (jnp.floor(x / node_size) + 0.5) * node_size
    return node_cent, node_size

def box_dist2(c1, c2, s1, s2, mode="shortest"):
    """Shortest or longest distance between any points in two nodes
    mode: "shortest" or "longest"
    """
    dx = jnp.abs(c1 - c2)
    if mode == "shortest":
        dx = jnp.maximum(dx - (s1 + s2) / 2., 0.)
    elif mode == "longest":
        dx = jnp.maximum(dx + (s1 + s2) / 2., 0.)
    else:
        raise ValueError("mode must be 'shortest' or 'longest'")
    return jnp.sum(dx**2, axis=-1)

def cumsum_starting_with_zero(x):
    return jnp.concatenate((jnp.zeros((1,) + x.shape[1:], dtype=x.dtype), jnp.cumsum(x, axis=0)))

def leaf_knn_estimates(xleaf, npart_leaf, level_leaf, bins=None, k=32):
    if bins is None:
        bins = jnp.logspace(-0.5, 1., 32)
    leaf_cent, leaf_ext = get_node_box(xleaf, level_leaf)
    nleaves = len(leaf_cent)

    def handle_single_leaf(ileaf):
        # We want to find the smallest distance at which we are guaranteed to find >= k neighbors
        # For this we need to include leaves whenever they exceed the distance where they are fully
        # included by every particle in the source leaf

        rbase2 = jnp.sum(leaf_ext[ileaf]**2, axis=-1)

        dist2 = box_dist2(leaf_cent[ileaf], leaf_cent, leaf_ext, leaf_ext, mode="longest")

        dratio2 = dist2 * (1./ rbase2)

        # bins, nkincl tells us how many neighbours are at least included at which distance:
        nkincl = jnp.cumsum(jnp.histogram(dratio2, bins=bins, weights=npart_leaf)[0])
        r2min = jnp.min(jnp.where(nkincl >= k, bins[1:], jnp.inf), axis=-1) * rbase2

        # To count the leaves we need to check at a given radius, we need to compare the closest
        # distance
        dist2min = box_dist2(leaf_cent[ileaf], leaf_cent, leaf_ext[ileaf], leaf_ext, mode="shortest")
        ninteractions = jnp.sum(dist2min <= r2min, axis=-1)

        return jnp.sqrt(r2min), ninteractions
    
    rneed, ninteractions = jax.vmap(handle_single_leaf)(jnp.arange(nleaves))

    offsets = cumsum_starting_with_zero(ninteractions)

    interactions = jnp.zeros(offsets[-1], dtype=jnp.int32)
    
    def insert_interactions(ileaf, interactions):
        dist2 = box_dist2(leaf_cent[ileaf], leaf_cent, leaf_ext[ileaf], leaf_ext, mode="shortest")
        dist2b = box_dist2(leaf_cent[ileaf], leaf_cent, leaf_ext[ileaf], leaf_ext, mode="longest")

        isort = jnp.lexsort([dist2b, dist2])

        iarange = jnp.arange(nleaves)
        ninleaf = offsets[ileaf + 1] - offsets[ileaf]
        ioff = jnp.where(iarange < ninleaf, offsets[ileaf] + iarange, offsets[-1])

        return interactions.at[ioff].set(isort)

    interactions = jax.lax.fori_loop(0, nleaves, insert_interactions, interactions)

    return rneed, interactions, offsets