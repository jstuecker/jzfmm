from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jztree.data import Pos, PosMass
from jztree.comm import get_rank_info, in_shard_map_context
from jztree.jax_ext import shard_map_constructor

from .config import DirectSummationConfig, FMMConfig
from .fmm import direct_summation, fast_multipole_method

def _as_posmass(part: PosMass | Pos | jax.Array) -> PosMass:
    if not hasattr(part, "pos"):
        pos = part
        mass = jnp.asarray(1.0 / len(pos), dtype=pos.dtype)
        return PosMass(pos=pos, mass=mass)

    pos = part.pos
    mass = getattr(part, "mass", None)
    if mass is None:
        mass = jnp.asarray(1.0 / len(pos), dtype=pos.dtype)

    return PosMass(
        pos=pos,
        mass=mass,
        num=getattr(part, "num", None),
        num_total=getattr(part, "num_total", None),
    )

def _mass_array(part: PosMass) -> jax.Array:
    mass = jnp.broadcast_to(part.mass, part.pos.shape[:-1])
    if part.num is not None:
        mass = jnp.where(jnp.arange(len(mass)) < part.num, mass, 0)
    return mass

def _normalize_mass(part: PosMass) -> PosMass:
    mass = _mass_array(part)
    mass_sum = jnp.sum(mass)
    if in_shard_map_context():
        _, _, axis_name = get_rank_info()
        mass_sum = jax.lax.psum(mass_sum, axis_name)
    return replace(part, mass=mass / mass_sum)

def _sum_num(a, b):
    if (a is None) or (b is None):
        return None
    return a + b

def _prepare_signed_particles(
    part: PosMass | Pos | jax.Array,
    part_target: PosMass | Pos | jax.Array,
    *,
    normalize: bool,
) -> PosMass:
    part = _as_posmass(part)
    part_target = _as_posmass(part_target)

    if normalize:
        part = _normalize_mass(part)
        part_target = _normalize_mass(part_target)

    mass = jnp.concatenate([
        _mass_array(part),
        -_mass_array(part_target),
    ])
    pos = jnp.concatenate([part.pos, part_target.pos], axis=0)
    return PosMass(
        pos=pos,
        mass=mass,
        num=_sum_num(part.num, part_target.num),
        num_total=_sum_num(part.num_total, part_target.num_total),
    )

def maximum_mean_discrepancy(
    part: PosMass | Pos | jax.Array,
    part_target: PosMass | Pos | jax.Array,
    *,
    cfg_fmm: FMMConfig | DirectSummationConfig,
    normalize: bool = True,
) -> jax.Array:
    """Maximum mean discrepancy between two weighted particle sets.

    The MMD is evaluated as a signed kernel energy, using positive masses for
    ``part`` and negative masses for ``part_target``: ``-sum_i m_i phi_i``.
    """
    particles = _prepare_signed_particles(part, part_target, normalize=normalize)
    dim = particles.pos.shape[-1]

    if isinstance(cfg_fmm, DirectSummationConfig):
        loc = direct_summation(particles, cfg_direct=cfg_fmm, G=1.0)
    elif isinstance(cfg_fmm, FMMConfig):
        if dim not in (2, 3):
            raise ValueError(
                "FMM MMD is only supported for dim=2 or dim=3. "
                "Use DirectSummationConfig(...) to use direct summation for dim=4-6."
            )
        loc = fast_multipole_method(particles, cfg_fmm=cfg_fmm, G=1.0)
    else:
        raise TypeError(f"Unsupported force config type {type(cfg_fmm)}")

    loss = -jnp.sum(particles.mass * loc.potential())
    if in_shard_map_context():
        _, _, axis_name = get_rank_info()
        loss = jax.lax.psum(loss, axis_name)
    return loss
maximum_mean_discrepancy.jit = jax.jit(
    maximum_mean_discrepancy,
    static_argnames=("cfg_fmm", "normalize"),
)
maximum_mean_discrepancy.smap = shard_map_constructor(
    maximum_mean_discrepancy,
    in_specs=(P(-1), P(-1), None, None),
    out_specs=P(),
    static_argnames=("cfg_fmm", "normalize"),
)
