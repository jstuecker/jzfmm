import fmdj
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
from fmdj.comm import all_to_all_with_irank

from fmdj_utils.ics import gaussian_blob

def pow2_upto(n: int) -> list[int]:
    out = []
    p = 1
    while p <= n:
        out.append(p)
        p *= 2
    return out

_MAX = jax.device_count()          # devices visible to this process
NDEVS = pow2_upto(_MAX)[::-1]

def get_mesh(ndev=-1):
    return jax.sharding.Mesh(jax.devices()[:ndev], ('gpus',), axis_types=(AxisType.Auto))

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.multi_gpu
def bench_all_to_all(jax_bench, ndev):
    Nperdev = 256**3

    def def_run(self_prob=0.,copy_self=True):
        @jax.shard_map(in_specs=P(), out_specs=P("gpus"), mesh=get_mesh(ndev))
        def run():
            rank = jax.lax.axis_index(axis_name="gpus")
            x = jax.random.uniform(jax.random.key(rank), Nperdev)
            xpad = jnp.pad(x, (0, int(Nperdev*0.2)))
            irank = jax.random.randint(jax.random.key(rank+113), xpad.shape, 0, ndev)
            irank = jnp.where(xpad <= self_prob, rank, irank)
            x, dev_spl = all_to_all_with_irank(irank, xpad, num=Nperdev, axis_name="gpus", copy_self=copy_self)
            return x
        return jax.jit(run)

    jb = jax_bench(jit_rounds=10, jit_loops=1, jit_warmup=2)

    for copy_self in True, False:
        prefix = 'copy_' if copy_self else ''
        jb.measure(fn_jit=def_run(0., copy_self), tag=f"{prefix}p0.0")
        jb.measure(fn_jit=def_run(0.5, copy_self), tag=f"{prefix}p0.5")
        jb.measure(fn_jit=def_run(0.9, copy_self), tag=f"{prefix}p0.9")
        jb.measure(fn_jit=def_run(0.99, copy_self), tag=f"{prefix}p0.99")
        jb.measure(fn_jit=def_run(1.0, copy_self), tag=f"{prefix}p1.0")

def smap_jit(f, ndev):
    fsm = jax.shard_map(f, in_specs=P(), out_specs=P("gpus"), mesh=get_mesh(ndev))
    return jax.jit(lambda x: fsm(x)) # the lambda helps with passing x as keyword argument

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
@pytest.mark.multi_gpu
def bench_meta_comm(jax_bench, ndev):
    Nperdev = 256**3
    axis_name = "gpus"

    def rng():
        rank = jax.lax.axis_index(axis_name)
        return jax.random.uniform(jax.random.key(rank), Nperdev)
    def get_sum(x):
        return jnp.sum((rng()+x)**2)
    
    jb = jax_bench(jit_rounds=10, jit_loops=5, jit_warmup=4)

    def nocomm(x):
        return get_sum(x).reshape(1)
    jb.measure(fn_jit=smap_jit(nocomm, ndev), x=0., tag="nocomm")

    def psum(x):
        return jax.lax.psum(get_sum(x), axis_name).reshape(-1)
    jb.measure(fn_jit=smap_jit(psum, ndev), x=0., tag="psum")

    def allgather(x):
        return jax.lax.all_gather(get_sum(x), axis_name).reshape(-1)
    jb.measure(fn_jit=smap_jit(allgather, ndev), x=0., tag="allgather")

    def all_to_all(x):
        val = get_sum(x) * jnp.arange(ndev)
        return jax.lax.all_to_all(val, axis_name, 0, 0, tiled=True).reshape(-1)
    jb.measure(fn_jit=smap_jit(all_to_all, ndev), x=0., tag="all_to_all")

    def ragged(x):
        val = get_sum(x) * jnp.arange(ndev)
        ones = jnp.ones(ndev, dtype=jnp.int32)
        devices = jnp.arange(ndev)
        out = jax.lax.ragged_all_to_all(val, val, devices, ones, devices, ones, axis_name=axis_name)
        return out
    jb.measure(fn_jit=smap_jit(ragged, ndev), x=0., tag="ragged")