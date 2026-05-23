import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import jax.numpy as jnp
import jax
import numpy as np
import pytest
from jztree_utils import ics

from fmdj.config import Config, FMMConfig, OpeningByAngle, PlummerKernel, QuarticPlummerKernel, Plummer2DKernel, SoftenedDistanceKernel
from fmdj.fmm import direct_summation, fast_multipole_method


def _device_array_bytes(x):
    return np.asarray(jax.block_until_ready(x)).tobytes()

@pytest.mark.parametrize("p", [2,3,4])
def test_fmm_uniform(p):
    cfg = Config(kernel=PlummerKernel(softening=0.05), fmm=FMMConfig(p=p, opening=OpeningByAngle(theta=0.4), kahan_summation=True))

    part = ics.uniform_particles(int(1e4))

    fref = direct_summation.jit(part, kernel=cfg.kernel, kahan=True, G=cfg.G()).force()
    ffmm = fast_multipole_method.jit(part, cfg=cfg).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    # print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
    assert jnp.median(ferr_rel) <= 2.*10**-(p+1)
    assert jnp.max(ferr_rel) <= 3.*10**-(p-1)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("dim", [2,3])
def test_dim(dim):
    p = 3
    cfg = Config(kernel=PlummerKernel(softening=0.05), fmm=FMMConfig(p=p, opening=OpeningByAngle(theta=0.4), kahan_summation=True))

    part = ics.uniform_particles(int(1e4), dim=dim)

    fref = direct_summation.jit(part, kernel=cfg.kernel, kahan=True, G=cfg.G()).force()
    ffmm = fast_multipole_method.jit(part, cfg=cfg).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    # print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
    assert jnp.median(ferr_rel) <= 8.*10**-(p+1)
    assert jnp.max(ferr_rel) <= 9.*10**-(p-1)

def test_gravity3d_kernels():
    cfg_plummer = Config(
        kernel=PlummerKernel(softening=1e-3),
        fmm=FMMConfig(p=3, opening=OpeningByAngle(theta=0.4), kahan_summation=True),
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


@pytest.mark.parametrize("kernel", [
    Plummer2DKernel(softening=0.05),
    SoftenedDistanceKernel(softening=0.05),
])
def test_other_kernels(kernel):
    cfg = Config(
        kernel=kernel,
        fmm=FMMConfig(p=4, opening=OpeningByAngle(theta=0.4), kahan_summation=True),
    )

    part = ics.uniform_particles(int(4096), dim=2)

    fref = direct_summation.jit(part, kernel=cfg.kernel, kahan=True, G=cfg.G()).force()
    ffmm = fast_multipole_method.jit(part, cfg=cfg).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    assert jnp.median(ferr_rel) <= 2e-4
    assert jnp.max(ferr_rel) <= 5e-2

def test_fmm_result_keys():
    cfg = Config(
        kernel=PlummerKernel(softening=0.05),
        fmm=FMMConfig(p=3, opening=OpeningByAngle(theta=0.4), kahan_summation=True),
    )
    part = ics.uniform_particles(2048)

    loc, partz, locz, th = fast_multipole_method.jit(
        part, cfg=cfg, result="loc_partz_locz_tree"
    )
    locz_from_tree = fast_multipole_method.jit(partz, cfg=cfg, th=th, result="locz")

    assert locz_from_tree.values == pytest.approx(locz.values)
    assert sorted(loc.potential().tolist()) == pytest.approx(sorted(locz.potential().tolist()))

    with pytest.raises(ValueError, match="result='loc'"):
        fast_multipole_method(partz, cfg=cfg, th=th, result="loc")


def test_fmm_reproducibility():
    cfg = Config()
    part = ics.uniform_particles(1024*1024)

    expected = _device_array_bytes(fast_multipole_method.jit(part, cfg=cfg).values)

    for _ in range(5):
        actual = _device_array_bytes(fast_multipole_method.jit(part, cfg=cfg).values)
        assert actual == expected
