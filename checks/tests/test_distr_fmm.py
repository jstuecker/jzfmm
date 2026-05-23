import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType

from jztree.data import squeeze_any, squeeze_particles
from jztree_utils import ics

from fmdj.config import Config
from fmdj.fmm import fast_multipole_method


mesh = jax.sharding.Mesh(jax.devices(), ("gpus",), axis_types=(AxisType.Auto,))


@pytest.mark.shrink_in_quick
def test_distr_fmm_matches_single_context():
    cfg = Config()
    cfg.tree.alloc_fac_nodes = 2.0

    part = ics.uniform_particles.smap(mesh, jit=True)(int(1e6), npad=int(4e5))

    partz, locz = fast_multipole_method.smap(mesh, jit=True)(part, cfg=cfg, result="partz_locz")
    locz = squeeze_any(locz.values, locz.values.shape[1], partz.num, partz.num_total)

    partz_flat = squeeze_particles(partz)
    loc_ref = fast_multipole_method.jit(partz_flat, cfg=cfg).values

    assert jnp.allclose(locz, loc_ref, rtol=1e-5, atol=1e-5)
