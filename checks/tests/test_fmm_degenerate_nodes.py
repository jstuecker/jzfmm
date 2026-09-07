"""Regression tests for sign-spanning and nearly zero-extent tree nodes."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jzfmm.config import DirectSummationConfig, FMMConfig, OpeningByAngle
from jzfmm.config import PlummerKernel, SoftenedDistanceKernel
from jzfmm.config import DKDLatticeConfig, SimConfig
from jzfmm.data import Particles, PosMass
from jzfmm.fmm import direct_summation, fast_multipole_method
from jzfmm.loss import maximum_mean_discrepancy
from jzfmm.time_integration import dequantize_particles, quantize_particles, simulate


def _positions(case, dtype, dim=3):
    rng = np.random.default_rng(14)
    pos = rng.uniform(1.0, 4.0, (2048, dim))
    if case == "sign_spanning":
        # A small isolated leaf crossing zero in the last coordinate.
        pos[:16] = rng.uniform(-1.0, 1.0, (16, dim))
        pos[:16, :dim-1] -= 3.0
    elif case == "coincident":
        pos = np.repeat(pos[::16], 16, axis=0)
    elif case == "tiny":
        pos[:128] = rng.uniform(-1.0, 1.0, (128, dim)) * 1e-20
    return jnp.asarray(pos, dtype=dtype)


# Quick mode retains each failure mechanism, covering both kernels in 3D/float32.
@pytest.mark.parametrize("case, distance_kernel", [
    ("sign_spanning", True),
    ("coincident", True),
    ("tiny", False),
    pytest.param("sign_spanning", False, marks=pytest.mark.skip_in_quick),
    pytest.param("coincident", False, marks=pytest.mark.skip_in_quick),
    pytest.param("tiny", True, marks=pytest.mark.skip_in_quick),
])
@pytest.mark.parametrize("dtype", [
    jnp.float32,
    pytest.param(jnp.float64, marks=pytest.mark.skip_in_quick),
])
@pytest.mark.parametrize("dim", [pytest.param(2, marks=pytest.mark.skip_in_quick), 3])
def test_degenerate_node_forces_and_gradients(case, distance_kernel, dtype, dim, order=5):
    with jax.enable_x64():
        pos = _positions(case, dtype, dim)
        mass = jnp.where(jnp.arange(len(pos)) % 2, -1.0, 1.0).astype(dtype) / len(pos)
        part = PosMass(pos=pos, mass=mass)
        kernel = (SoftenedDistanceKernel if distance_kernel else PlummerKernel)(softening=0.3)
        cfg = FMMConfig(kernel=kernel, p=order, opening=OpeningByAngle(theta=0.3),
                        alloc_fac_ilist=256., remove_self_interaction=False)
        direct_cfg = DirectSummationConfig(kernel=kernel, remove_self_interaction=False)

        def evaluate(p, direct):
            if direct:
                return direct_summation(p, cfg_direct=direct_cfg).values
            return fast_multipole_method(p, cfg_fmm=cfg).values

        ref = jax.jit(lambda p: evaluate(p, True))(part)
        actual = jax.jit(lambda p: evaluate(p, False))(part)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, ref, atol=2e-5, rtol=2e-3)

        def energy(p, direct):
            return jnp.sum(p.mass * evaluate(p, direct)[:, 0])

        grad_ref = jax.jit(jax.grad(lambda p: energy(p, True)))(part)
        grad = jax.jit(jax.grad(lambda p: energy(p, False)))(part)
        for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(grad_ref)):
            assert np.isfinite(a).all()
            np.testing.assert_allclose(a, b, atol=2e-5, rtol=3e-3)


@pytest.mark.skip_in_quick
@pytest.mark.parametrize("case", ["sign_spanning", "coincident", "tiny"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_high_order_degenerate_nodes(case, dtype):
    test_degenerate_node_forces_and_gradients(case, True, dtype, dim=3, order=7)


@pytest.mark.skip_in_quick
def test_coincident_mmd_gradient():
    pos = _positions("coincident", jnp.float32)
    part = PosMass(pos=pos, mass=jnp.full(len(pos), 1.0 / len(pos)))
    cfg = FMMConfig(kernel=SoftenedDistanceKernel(softening=0.3),
                    remove_self_interaction=False)

    def loss(shift):
        return maximum_mean_discrepancy(replace(part, pos=part.pos + shift), part, cfg_fmm=cfg)

    value, grad = jax.jit(jax.value_and_grad(loss))(jnp.zeros(3))
    assert np.isfinite(value) and np.isfinite(grad).all()
    np.testing.assert_allclose(value, 0.0, atol=1e-5)
    np.testing.assert_allclose(grad, 0.0, atol=1e-5)


def test_sign_spanning_simulation_reversibility():
    pos = _positions("sign_spanning", jnp.float32)
    part = Particles(pos=pos, mass=jnp.full(len(pos), 1.0 / len(pos)),
                     vel=jnp.full_like(pos, 0.1))
    cfg = SimConfig(integrator=DKDLatticeConfig(dx=1e-5, dv=1e-5))
    part = dequantize_particles(quantize_particles(part, cfg.integrator), cfg.integrator)
    ts = jnp.linspace(0.0, 0.1, 5)
    final = simulate.jit(part, ts, cfg)
    back = simulate.jit(final, ts[::-1], cfg)
    np.testing.assert_array_equal(back.pos, part.pos)
    np.testing.assert_array_equal(back.vel, part.vel)
