"""GPU regression checks for the experimental softened-angle criterion."""
from dataclasses import replace
import jax
import numpy as np
from jztree_utils import ics
from jzfmm.config import FMMConfig, OpeningBySoftenedAngle, PlummerKernel
from jzfmm.fmm import fast_multipole_method, _evaluate_node_node_fmm


def test_zero_softening_recovers_geometric_walk():
    part = ics.uniform_particles(8192, seed=7)
    cfg = FMMConfig()
    partz, tree = fast_multipole_method.jit(part, cfg, result='partz_tree')
    softened = replace(cfg, opening=OpeningBySoftenedAngle(softening=0.))
    original = fast_multipole_method.jit(partz, cfg, th=tree, result='locz')
    actual = fast_multipole_method.jit(partz, softened, th=tree, result='locz')
    np.testing.assert_array_equal(np.asarray(actual.values), np.asarray(original.values))
    _, old_list = _evaluate_node_node_fmm.jit(partz, tree, cfg_fmm=cfg)
    _, new_list = _evaluate_node_node_fmm.jit(partz, tree, cfg_fmm=softened)
    np.testing.assert_array_equal(np.asarray(old_list.ispl), np.asarray(new_list.ispl))
    n = int(old_list.nfilled())
    np.testing.assert_array_equal(np.asarray(old_list.isrc[:n]), np.asarray(new_list.isrc[:n]))


def test_softened_opening_reduces_near_field_list():
    part = ics.uniform_particles(8192, seed=7)
    eps = 3 * 8192**(-1/3)
    cfg = FMMConfig(kernel=PlummerKernel(softening=eps))
    partz, tree = fast_multipole_method.jit(part, cfg, result='partz_tree')
    softened = replace(cfg, opening=OpeningBySoftenedAngle(softening=eps))
    _, old_list = _evaluate_node_node_fmm.jit(partz, tree, cfg_fmm=cfg)
    _, new_list = _evaluate_node_node_fmm.jit(partz, tree, cfg_fmm=softened)
    assert int(new_list.nfilled()) < int(old_list.nfilled())
    actual = fast_multipole_method.jit(partz, softened, th=tree, result='locz')
    assert np.isfinite(np.asarray(actual.values)).all()
