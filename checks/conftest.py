import jax
import jax.numpy as jnp
import pytest
import fmdj

def get_particles(N = 1024*1024):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
    pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
    mass = jnp.ones(N, dtype=jnp.float32)
    
    return jax.block_until_ready((pos0, mass))

@pytest.fixture
def particles(request):
    Npart = getattr(request, "param", 1024*1024)
    
    return get_particles(Npart)

@pytest.fixture
def tree(particles, request):
    pos0, mass0 = particles
    p = getattr(request, "param", 2)
    octree, posz, massz, isortz = fmdj.fmm.build_octree_with_multipoles.jit(pos0+0.5, mass0, p=p)

    return octree, posz, massz, isortz

@pytest.fixture
def interactions(tree, request):
    octree, posz, massz, isortz = tree
    ilist, nilist = fmdj.fmm.build_interaction_list.jit(octree)
    ilist, iranges = fmdj.fmm.organize_interactions.jit(ilist, nilist, sort=False)
    ilist.block_until_ready()
    return octree, posz, massz, ilist, nilist, iranges