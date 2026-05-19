import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import jax.numpy as jnp
import jax
import pytest
from jztree_utils import ics

from fmdj.config import Config, FMMConfig, PlummerKernel, QuarticPlummerKernel
from fmdj.data import PosMass, LocalExpansion
from fmdj.fmm import direct_force_and_potential, fast_multipole_method

@pytest.mark.parametrize("p", [2,3,4])
def test_fmm_uniform(p):
    cfg = Config(kernel=PlummerKernel(softening=0.05), fmm=FMMConfig(p=p, opening_angle=0.4, kahan_summation=True))

    part = ics.uniform_particles(int(1e4))

    fref = LocalExpansion(direct_force_and_potential.jit(part, kernel=cfg.kernel, kahan=True) * cfg.G()).force()
    ffmm = fast_multipole_method.jit(part, cfg=cfg).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    # print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
    assert jnp.median(ferr_rel) <= 10**-(p+1)
    assert jnp.max(ferr_rel) <= 10**-(p-1)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("dim", [2,3])
def test_dim(dim):
    p = 3
    cfg = Config(kernel=PlummerKernel(softening=0.05), fmm=FMMConfig(p=p, opening_angle=0.4, kahan_summation=True))

    part = ics.uniform_particles(int(1e4), dim=dim)

    fref = LocalExpansion(
        values=direct_force_and_potential.jit(part, kernel=cfg.kernel, kahan=True) * cfg.G(),
        dim = dim
    ).force()
    ffmm = fast_multipole_method.jit(part, cfg=cfg).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    # print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
    assert jnp.median(ferr_rel) <= 2.*10**-(p+1)
    assert jnp.max(ferr_rel) <= 5.*10**-(p-1)

def test_softening_kernels():
    cfg_plummer = Config(
        kernel=PlummerKernel(softening=1e-3),
        fmm=FMMConfig(p=3, opening_angle=0.4, kahan_summation=True),
    )
    cfg_quartic = Config(
        kernel=QuarticPlummerKernel(softening=1e-3),
        fmm=cfg_plummer.fmm,
    )

    part = ics.uniform_particles(int(2048))

    f_plummer = fast_multipole_method.jit(part, cfg=cfg_plummer).force()
    f_quartic = fast_multipole_method.jit(part, cfg=cfg_quartic).force()

    ferr = jnp.linalg.norm(f_quartic - f_plummer, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(f_quartic + f_plummer), axis=-1)

    assert jnp.median(ferr_rel) <= 1e-3
    assert jnp.max(ferr_rel) <= 5e-2
