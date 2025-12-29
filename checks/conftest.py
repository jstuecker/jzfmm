import jax
import jax.numpy as jnp
import pytest
import fmdj
import os
import sys

def _silence_process_output() -> None:
    """
    Redirect stdout/stderr to /dev/null at the OS FD level.
    Also rebind sys.stdout/sys.stderr to avoid some Python-level oddities.
    """
    # Make Python less likely to buffer weirdly
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # Redirect low-level file descriptors (covers most output sources)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)  # stdout
    os.dup2(devnull_fd, 2)  # stderr
    os.close(devnull_fd)

    # Rebind Python-level streams (some libs write to these objects directly)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

def pytest_configure(config):
    try:
        jax.distributed.initialize()
    except ValueError as err:
        print(f"Distributed mode not available ({err})")

    if jax.process_index() != 0:
        _silence_process_output()

def get_particles(N = 1024*1024):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.3
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
    th = jax.block_until_ready(fmdj.fmm.build_tree_hierarchy.jit(pos_mass_z, cfg=cfg))
    return th

@pytest.fixture
def tree_planes(tree_hierarchy):
    return list(tree_hierarchy.planes())

@pytest.fixture
def particles_blob(npart):
    x = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    m = jnp.ones_like(x[:,0]) * 1.
    vel = jnp.zeros_like(x)

    loc = fmdj.data.LocalExpansion(jnp.zeros((npart,4), dtype=jnp.float32))

    return fmdj.data.Particles(x, m, vel, cpos=jnp.array([0.,0.,0.]), cvel=jnp.array([0.,0.,0.0]), loc=loc)

@pytest.fixture
def particles_nfw(npart):
    import aegis
    prof = aegis.profiles.NFWProfile(conc=10., r200c=10.)
    pos0, vel0, m = prof.sample_particles(npart, result="pos_vel_m", rpmin=1e-3, ramax=10.)

    part = fmdj.data.Particles(jnp.array(pos0), jnp.array(m), jnp.array(vel0))
    part.cpos = jnp.array((150.,0.,0.))
    part.cvel = jnp.array((0.,prof.vcirc(150.),0.))
    part.loc = fmdj.data.LocalExpansion(jnp.zeros((npart,4), dtype=jnp.float32))

    return part

# ------------------------------------------------------------------------------------------------ #
#                                       Multi GPU specific                                         #
# ------------------------------------------------------------------------------------------------ #

def pytest_report_header(config):
    # Show once per pytest run (rank 0 only)
    if jax.process_index() != 0:
        return
    return [
        f"JAX processes: {getattr(jax, 'process_count', lambda: 1)()}",
        f"JAX device_count: {jax.device_count()}",
        f"JAX local_device_count: {jax.local_device_count()}",
    ]

def pytest_unconfigure(config):
    """The final cleanup."""
    try:
        if hasattr(jax.distributed, 'shutdown'):
            jax.distributed.shutdown()
    except:
        pass