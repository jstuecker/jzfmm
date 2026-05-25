import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, PartitionSpec as P

from jztree.data import squeeze_any, squeeze_particles
from jztree.jax_ext import shard_map_constructor
from jztree.comm import get_rank_info, in_shard_map_context
from jztree_utils import ics

from fmdj.config import FMMConfig
from fmdj.fmm import fast_multipole_method


mesh = jax.sharding.Mesh(jax.devices(), ("gpus",), axis_types=(AxisType.Auto,))


def _fmm_loss(part, cfg_fmm):
    loc = fast_multipole_method(part, cfg_fmm=cfg_fmm, result="loc").values
    valid = jnp.arange(loc.shape[0]) < part.num
    loss = jnp.sum(jnp.where(valid[:, None], loc, 0.0))
    if in_shard_map_context():
        _, _, axis_name = get_rank_info()
        loss = jax.lax.psum(loss, axis_name)
    return loss


def _fmm_grad(part, cfg_fmm):
    return jax.grad(_fmm_loss, allow_int=True)(part, cfg_fmm)


_fmm_grad.smap = shard_map_constructor(
    _fmm_grad,
    in_specs=(P(-1), None),
    out_specs=P(-1),
    static_argnames=("cfg_fmm",),
)

def test_distr_vs_single():
    # This test checks for bit-perfect reproducibility of fmm accross GPU counts
    cfg_fmm = FMMConfig()
    cfg_fmm.tree.alloc_fac_nodes = 2.0

    part = ics.uniform_particles.smap(mesh, jit=True)(int(1e6), npad=int(4e5))

    partz, locz = fast_multipole_method.smap(mesh, jit=True)(part, cfg_fmm=cfg_fmm, result="partz_locz")
    locz = squeeze_any(locz.values, locz.values.shape[1], partz.num, partz.num_total)

    partz_flat = squeeze_particles(partz)
    loc_ref = fast_multipole_method.jit(partz_flat, cfg_fmm=cfg_fmm).values

    assert jnp.all(locz == loc_ref)

def test_distr_grad_vs_single():
    # Check for bit-perfect reproducibility of gradients accross GPU counts
    cfg_fmm = FMMConfig()
    cfg_fmm.tree.alloc_fac_nodes = 2.0

    part = ics.uniform_particles.smap(mesh, jit=True)(32768, npad=8192*2)
    part = replace(part, mass=jnp.broadcast_to(part.mass[:, None], part.pos.shape[:-1]))

    grad = _fmm_grad.smap(mesh, jit=True)(part, cfg_fmm)
    grad = replace(grad, num=part.num, num_total=part.num_total)
    grad = squeeze_particles(grad)

    part_flat = squeeze_particles(part)
    grad_ref = jax.jit(_fmm_grad, static_argnames=("cfg_fmm",))(part_flat, cfg_fmm)

    assert jnp.all(grad.pos == grad_ref.pos)
    assert jnp.all(grad.mass == grad_ref.mass)
