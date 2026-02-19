import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import jax.numpy as jnp
import jax
import pytest
from jztree_utils import ics

from fmdj.config import Config, FMMConfig
from fmdj.data import PosMass, LocalExpansion
from fmdj.fmm import direct_force_and_potential, fast_multipole_method

@pytest.mark.parametrize("p", [2,3,4])
def test_fmm_uniform(p):
    cfg = Config(softening=0.05, fmm=FMMConfig(p=p, opening_angle=0.4, kahan_summation=True))

    part = ics.uniform_particles(int(1e4))

    fref = LocalExpansion(direct_force_and_potential.jit(part, softening=cfg.softening, kahan=True) * cfg.G()).force()
    ffmm = fast_multipole_method.jit(part, cfg=cfg).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    # print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
    assert jnp.median(ferr_rel) <= 10**-(p+1)
    assert jnp.max(ferr_rel) <= 10**-(p-1)

# @pytest.mark.parametrize("p", [2,3,4])
# def test_mem_err(p):
#     """Currently creates some memory access error, need to figure this one out!"""
#     cfg = Config(softening=0.05, fmm=FMMConfig(p=p, opening_angle=0.4, kahan_summation=True))

#     part = ics.uniform_particles(int(1e5))

#     fref = LocalExpansion(direct_force_and_potential.jit(part, softening=cfg.softening, kahan=True) * cfg.G()).force()
#     ffmm = fast_multipole_method.jit(part, cfg=cfg).force()

#     ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
#     ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

#     print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
#     assert jnp.median(ferr_rel) <= 10**-(p+1)
#     assert jnp.max(ferr_rel) <= 10**-(p-1)