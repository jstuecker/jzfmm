import fmdj
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))

ndev = len(jax.devices())

@pytest.mark.multi_gpu
def test_mutli_zsort():
    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=mesh)
    def myzsort():
        rank = jax.lax.axis_index(axis_name="gpus")
        pos = jax.random.uniform(jax.random.key(rank), (int(4096+1024),3))
        pos = pos.at[-1024:].set(jnp.nan)

        posz = fmdj.ztree.distributed_zsort(pos, fmdj.config.CommunicationConfig())
        return pos, posz

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