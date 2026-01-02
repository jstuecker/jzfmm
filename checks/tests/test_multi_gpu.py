import fmdj
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))

ndev = len(jax.devices())

@jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=mesh)
def myzsort():
    rank = jax.lax.axis_index(axis_name="gpus")
    pos = jax.random.uniform(jax.random.key(rank), (int(4096+1024),3))
    pos = pos.at[-1024:].set(jnp.nan)

    posz = fmdj.ztree.distributed_zsort(pos, fmdj.config.CommunicationConfig())
    return pos, posz

@pytest.mark.multi_gpu
def test_mutli_zsort():
    pos, posz = jax.jit(myzsort)()
    
    shard0 = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    p0 = jax.device_put(pos, shard0)
    
    posz0a = fmdj.ztree.pos_zorder_sort(p0)[0] # single device sort
    posz0b = jax.device_put(posz, shard0)      # multi device sort transferred to single device

    posz0a = posz0a[~jnp.isnan(posz0a[:,0])]
    posz0b = posz0b[~jnp.isnan(posz0b[:,0])]

    print(f"Testing sort with {len(jax.devices())} devices.")

    assert len(posz0a) == len(posz0b)
    assert jnp.all(posz0a == posz0b)

@jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=mesh)
def fcoarsen(posz: jnp.ndarray):
    # return fmdj.ztree.distributed_define_leaves(posz, leaf_size=256)
    posz, npart = fmdj.ztree.adjust_domain_for_nodesize(posz, 256)
    ispl = fmdj.ztree.create_coarse_leaves(posz, leaf_size=256)

    return posz, ispl

@jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=mesh)
def splits_to_global(ispl):
    """Adds offsets to locally calculated splits so they correspond to global array splits"""
    from fmdj.comm import get_rank_info
    rank, ndev, axis_name = get_rank_info()

    neach = jax.lax.all_gather(ispl[-1], axis_name)
    offset = jnp.cumsum(neach)[rank] - ispl[-1]
    ispl = ispl + offset

    return ispl

@pytest.mark.multi_gpu
def test_multi_leaves():
    pos, posz = jax.jit(myzsort)()
    posz_new, ispln = jax.jit(fcoarsen)(posz)
    
    # Check that position array was not messed up
    def remove_invalid(x):
        return x[jnp.where(~jnp.isnan(x[...,0]))]
    
    assert jnp.all(remove_invalid(posz) == remove_invalid(posz_new))

    # Check splits against locally calculated ones
    posz0 = jax.device_put(posz, jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    ispl0 = fmdj.ztree.create_coarse_leaves(remove_invalid(posz0), leaf_size=256)

    def remove_duplicates(i):
        return i[jnp.where(i[1:] != i[:-1])]
    
    ispln2 = splits_to_global(ispln)
    ispln2 = jax.device_put(ispln2, jax.sharding.SingleDeviceSharding(jax.devices()[0]))

    assert jnp.all(remove_duplicates(ispln2) == remove_duplicates(ispl0))