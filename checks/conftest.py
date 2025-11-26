import jax
import jax.numpy as jnp
import pytest
import fmdj
import aegis

def get_particles(N = 1024*1024):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.1
    pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
    mass = jnp.ones(N, dtype=jnp.float32)
    
    return jax.block_until_ready((pos0, mass))

# ------------------------------------------------------------------------------------------------ #
#                                             Fixtures                                             #
# ------------------------------------------------------------------------------------------------ #

@pytest.fixture
def cfg():
    return fmdj.Config()

@pytest.fixture
def npart(request):    
    return getattr(request, "param", 1024*1024)

@pytest.fixture
def pos_mass(npart):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    return fmdj.data.PosMass(pos0, mass=jnp.ones(pos0.shape[0]))

@pytest.fixture
def pos_mass_z(npart):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    posz, isort = fmdj.ztree.pos_zorder_sort(pos0)
    return fmdj.data.PosMass(posz, jnp.ones(posz.shape[0]))

@pytest.fixture
def tree_hierarchy(pos_mass_z, cfg):
    ths : list[fmdj.data.TreePlane] = jax.block_until_ready(fmdj.fmm.build_tree_hierarchy.jit(pos_mass_z, cfg=cfg))
    return ths

@pytest.fixture
def particles_blob(npart):
    x = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    m = jnp.ones_like(x[:,0]) * 1.
    vel = jnp.zeros_like(x)

    return fmdj.data.Particles(x, vel, m, cpos=jnp.array([0.,0.,0.]), cvel=jnp.array([0.,0.,0.0]))

@pytest.fixture
def particles_nfw(npart):
    prof = aegis.profiles.NFWProfile(conc=10., r200c=10.)
    pos0, vel0, m = prof.sample_particles(npart, result="pos_vel_m", rpmin=1e-3, ramax=10.)

    part = fmdj.data.Particles(jnp.array(pos0), jnp.array(vel0), jnp.array(m))
    part.cpos = jnp.array((150.,0.,0.))
    part.cvel = jnp.array((0.,prof.vcirc(150.),0.))

    return part