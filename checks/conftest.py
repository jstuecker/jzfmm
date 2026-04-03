import jax
import jax.numpy as jnp
import pytest
import os
import sys
from jztree.comm import should_init_jax_distributed
from fmdj import Config
from fmdj.data import Particles, PosMass, LocalExpansion
from jztree.tree import zsort, build_tree_hierarchy

# ------------------------------------------------------------------------------------------------ #
#                                         Configure pytest                                         #
# ------------------------------------------------------------------------------------------------ #

def pytest_addoption(parser):
    parser.addoption( "--quick", action="store_true", default=False,
        help="Quick mode: deselect slow tests and reduce parametrized tests to first case.",
    )

def pytest_runtest_setup(item):
    if item.config.getoption("--quick"):
        if item.get_closest_marker("skip_in_quick") or item.get_closest_marker("slow"):
            pytest.skip("Skipped in --quick mode")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: skip the whole test in --quick mode")
    config.addinivalue_line("markers", "skip_in_quick: skip the whole test in --quick mode")
    config.addinivalue_line( "markers",
        "shrink_in_quick(keep_index=0): in --quick mode, keep only the parametrized at index",
    )
    # Your existing setup:
    if should_init_jax_distributed():
        jax.distributed.initialize()
    else:
        print("Using single-host mode")

    if jax.process_index() != 0:
        _silence_process_output()

def _keep_index_from_marker(item) -> int | None:
    """
    Returns the per-test keep index from @pytest.mark.shrink_in_quick(index=...)
    (or @pytest.mark.shrink_in_quick(<int>)), or None if marker not present.
    """
    m = item.get_closest_marker("shrink_in_quick")
    if m is None:
        return None

    if "keep_index" in m.kwargs:
        return int(m.kwargs["keep_index"])
    if m.args:
        return int(m.args[0])
    return 0

def pytest_runtest_setup(item):
    if not item.config.getoption("--quick"):
        return

    # Skip whole tests marked slow
    if item.get_closest_marker("slow") or item.get_closest_marker("skip_in_quick"):
        pytest.skip("Skipped in --quick mode (slow / skip_in_quick)")

    # Shrink parametrized tests that opt in
    keep = _keep_index_from_marker(item)
    if keep is None:
        return

    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return  # not parametrized

    indices = getattr(callspec, "indices", {}) or {}
    if indices and any(i != keep for i in indices.values()):
        pytest.skip(f"Skipped in --quick mode (keep={keep})")

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


# ------------------------------------------------------------------------------------------------ #
#                                             Fixtures                                             #
# ------------------------------------------------------------------------------------------------ #

def get_particles(N = 1024*1024):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (N, 3), dtype=jnp.float32) * 0.3
    pos0 = jnp.clip(pos0, -0.5, 0.5).block_until_ready()
    mass = jnp.ones(N, dtype=jnp.float32)
    
    return jax.block_until_ready((pos0, mass))

@pytest.fixture
def cfg():
    return Config()

@pytest.fixture
def npart(request):    
    return getattr(request, "param", 1024*1024)

@pytest.fixture
def pos_mass(npart):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    return PosMass(pos=pos0, mass=jnp.ones(pos0.shape[0]))

@pytest.fixture
def pos_mass_z(npart):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    posz, isort = zsort(pos0)
    return PosMass(pos=posz, mass=jnp.ones(posz.shape[0]))

@pytest.fixture
def tree_hierarchy(pos_mass_z, cfg):
    th = jax.block_until_ready(build_tree_hierarchy.jit(pos_mass_z, cfg_tree=cfg.tree))
    return th

@pytest.fixture
def particles_blob(npart):
    x = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    m = jnp.ones_like(x[:,0]) * 1.
    vel = jnp.zeros_like(x)

    loc = LocalExpansion(jnp.zeros((npart,4), dtype=jnp.float32))

    return Particles(pos=x, mass=m, vel=vel, cpos=jnp.array([0.,0.,0.]), cvel=jnp.array([0.,0.,0.0]), loc=loc)

@pytest.fixture
def particles_nfw(npart):
    import aegis
    prof = aegis.profiles.NFWProfile(conc=10., r200c=10.)
    pos0, vel0, m = prof.sample_particles(npart, result="pos_vel_m", rpmin=1e-3, ramax=10.)

    part = Particles(pos=jnp.array(pos0), mass=jnp.array(m), vel=jnp.array(vel0))
    part.cpos = jnp.array((150.,0.,0.))
    part.cvel = jnp.array((0.,prof.vcirc(150.),0.))
    part.loc = LocalExpansion(jnp.zeros((npart,4), dtype=jnp.float32))

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