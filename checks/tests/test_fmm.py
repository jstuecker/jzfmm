import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import jax.numpy as jnp
import jax
import numpy as np
import pytest
from jztree_utils import ics

from jzfmm.config import DirectSummationConfig, FMMConfig, OpeningByAngle, PlummerKernel, Plummer2DKernel, SoftenedDistanceKernel
from jzfmm.fmm import direct_summation, fast_multipole_method

def _device_array_bytes(x):
    return np.asarray(jax.block_until_ready(x)).tobytes()

def _cfg_direct(cfg_fmm: FMMConfig) -> DirectSummationConfig:
    return DirectSummationConfig(kernel=cfg_fmm.kernel, kahan_summation=True)

@pytest.mark.parametrize("p", [2,3,4])
def test_fmm_uniform(p):
    cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=0.05), p=p, opening=OpeningByAngle(theta=0.4), kahan_summation=True)
    cfg_fmm.alloc_fac_ilist = 256. # need larger, because small opening angle

    part = ics.uniform_particles(int(1e4))

    fref = direct_summation.jit(part, cfg_direct=_cfg_direct(cfg_fmm)).force()
    ffmm = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    # print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
    assert jnp.median(ferr_rel) <= 2.*10**-(p+1)
    assert jnp.max(ferr_rel) <= 3.*10**-(p-1)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("dim", [2,3])
def test_dim(dim):
    p = 3
    cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=0.05), p=p, opening=OpeningByAngle(theta=0.4), kahan_summation=True)

    part = ics.uniform_particles(int(1e4), dim=dim)

    fref = direct_summation.jit(part, cfg_direct=_cfg_direct(cfg_fmm)).force()
    ffmm = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    # print(p, jnp.median(ferr_rel) / 10**(-p-1), jnp.max(ferr_rel) / 10**(1-p))
    assert jnp.median(ferr_rel) <= 8.*10**-(p+1)
    assert jnp.max(ferr_rel) <= 9.*10**-(p-1)


@pytest.mark.parametrize("kernel", [
    Plummer2DKernel(softening=0.05),
    SoftenedDistanceKernel(softening=0.05),
])
def test_other_kernels(kernel):
    cfg_fmm = FMMConfig(
        kernel=kernel,
        p=4,
        opening=OpeningByAngle(theta=0.4),
        kahan_summation=True,
    )

    part = ics.uniform_particles(int(4096), dim=2)

    fref = direct_summation.jit(part, cfg_direct=_cfg_direct(cfg_fmm)).force()
    ffmm = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).force()

    ferr = jnp.linalg.norm(fref - ffmm, axis=-1)
    ferr_rel = ferr / jnp.linalg.norm(0.5*(fref + ffmm), axis=-1)

    assert jnp.median(ferr_rel) <= 2e-4
    assert jnp.max(ferr_rel) <= 5e-2

def test_fmm_result_keys():
    cfg_fmm = FMMConfig(
        kernel=PlummerKernel(softening=0.05),
        p=3,
        opening=OpeningByAngle(theta=0.4),
        kahan_summation=True,
    )
    part = ics.uniform_particles(2048)

    loc, partz, locz, th = fast_multipole_method.jit(
        part, cfg_fmm=cfg_fmm, result="loc_partz_locz_tree"
    )
    locz_from_tree = fast_multipole_method.jit(partz, cfg_fmm=cfg_fmm, th=th, result="locz")

    assert locz_from_tree.values == pytest.approx(locz.values)
    assert sorted(loc.potential().tolist()) == pytest.approx(sorted(locz.potential().tolist()))

    with pytest.raises(ValueError, match="result='loc'"):
        fast_multipole_method(partz, cfg_fmm=cfg_fmm, th=th, result="loc")

def test_padding():
    cfg_fmm = FMMConfig()
    npart = 2048
    part = ics.uniform_particles(npart)
    part_padded = ics.uniform_particles(npart, npad=512)

    loc = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values
    loc_padded = fast_multipole_method.jit(part_padded, cfg_fmm=cfg_fmm).values

    assert jnp.all(loc_padded[:npart] == loc)
    assert jnp.all(jnp.isnan(loc_padded[npart:]))

def test_fmm_reproducibility():
    cfg_fmm = FMMConfig()
    part = ics.uniform_particles(1024*1024)

    expected = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values

    for _ in range(5):
        actual = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values
        assert jnp.all(expected == actual) # check bit-perfect agreement

@pytest.mark.parametrize("logscale", [-28,-16,-8,0,8,16,28])
def test_fmm_scales(logscale):
    cfg_fmm = FMMConfig()
    part = ics.gaussian_particles(1024*1024, scale=10.**logscale)

    loc = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values

    assert jnp.all(~jnp.isnan(loc))
