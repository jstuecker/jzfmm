from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P

from jztree.data import squeeze_particles
from jztree.jax_ext import shard_map_constructor
from jztree_utils import ics

from jzfmm.config import DKDConfig, DKDLatticeConfig, FMMConfig, SimConfig
from jzfmm.data import Particles
from jzfmm.time_integration import dequantize_particles, quantize_particles, simulate


mesh = jax.sharding.Mesh(jax.devices(), ("gpus",), axis_types=(AxisType.Auto,))


def _gaussian_particles(pos_mass):
    key_pos, key_vel = jax.random.split(jax.random.PRNGKey(0))
    valid = jnp.arange(pos_mass.pos.shape[1]) < pos_mass.num[:, None]
    pos = 0.3 * jax.random.normal(key_pos, pos_mass.pos.shape, dtype=pos_mass.pos.dtype)
    vel = 0.1 * jax.random.normal(key_vel, pos_mass.pos.shape, dtype=pos_mass.pos.dtype)
    return Particles(
        pos=jnp.where(valid[..., None], pos, pos_mass.pos),
        vel=jnp.where(valid[..., None], vel, 0.0),
        mass=jnp.broadcast_to(pos_mass.mass[:, None], pos_mass.pos.shape[:-1]),
        num=pos_mass.num,
        num_total=pos_mass.num_total,
    )


def _simulate(part, ts, cfg):
    return simulate(part, ts=ts, cfg=cfg)


_simulate.smap = shard_map_constructor(
    _simulate,
    in_specs=(P(-1), None, None),
    out_specs=P(-1),
    static_argnames=("cfg",),
)


def _loss(part, ts, cfg):
    part_final = simulate(part, ts=ts, cfg=cfg)
    valid = jnp.arange(part_final.pos.shape[0]) < part_final.num
    loss = jnp.sum(jnp.where(valid[:, None], part_final.pos**2, 0.0))
    return jax.lax.psum(loss, "gpus")


def _gradient(part, ts, cfg):
    return jax.grad(_loss, allow_int=True)(part, ts, cfg)


_gradient.smap = shard_map_constructor(
    _gradient,
    in_specs=(P(-1), None, None),
    out_specs=P(-1),
    static_argnames=("cfg",),
)


def test_distr_sim_reversibility():
    cfg = SimConfig(
        force=FMMConfig(),
        integrator=DKDLatticeConfig(dx=1e-4, dv=1e-4, int_dtype=jnp.int32),
    )
    pos_mass = ics.uniform_particles.smap(mesh, jit=True)(32768, npad=16384)
    part = _gaussian_particles(pos_mass)
    part = dequantize_particles(quantize_particles(part, cfg.integrator), cfg.integrator)
    ts = jnp.linspace(0.0, 0.01, 3, dtype=part.pos.dtype)

    part_final = _simulate.smap(mesh, jit=True)(part, ts, cfg)
    part_back = _simulate.smap(mesh, jit=True)(part_final, ts[::-1], cfg)

    assert jnp.array_equal(part_back.pos, part.pos)
    assert jnp.array_equal(part_back.vel, part.vel)


def test_distr_sim_gradients():
    cfg = SimConfig(force=FMMConfig(), integrator=DKDConfig())
    pos_mass = ics.uniform_particles.smap(mesh, jit=True)(32768, npad=16384)
    part = _gaussian_particles(pos_mass)
    ts = jnp.linspace(0.0, 0.01, 3, dtype=part.pos.dtype)

    grad = _gradient.smap(mesh, jit=True)(part, ts, cfg)
    grad = squeeze_particles(replace(grad, num=part.num, num_total=part.num_total))

    part_flat = squeeze_particles(part)

    def loss_reference(p):
        return jnp.sum(simulate(p, ts=ts, cfg=cfg).pos**2)

    grad_reference = jax.jit(jax.grad(loss_reference, allow_int=True))(part_flat)

    assert jnp.array_equal(grad.pos, grad_reference.pos)
    assert jnp.array_equal(grad.vel, grad_reference.vel)
    assert jnp.array_equal(grad.mass, grad_reference.mass)
