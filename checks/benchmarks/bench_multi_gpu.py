import fmdj
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

def pow2_upto(n: int) -> list[int]:
    out = []
    p = 1
    while p <= n:
        out.append(p)
        p *= 2
    return out

_MAX = jax.device_count()          # devices visible to this process
NDEVS = pow2_upto(_MAX)

def mk_pos(Ntot, alloc_fac=1.2, ndev=4):
    N = Ntot // ndev
    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=get_mesh(ndev))
    def f():
        rank = jax.lax.axis_index(axis_name="gpus")
        Nextra = int(N*(alloc_fac-1.))
        pos = jax.random.normal(jax.random.key(rank), (int(N+Nextra),3))
        pos = pos.at[-Nextra:].set(jnp.nan)

        return pos

    return f

def mksort(ndev=4):
    @jax.shard_map(out_specs=P("gpus"), in_specs=P("gpus"), mesh=get_mesh(ndev))
    def f(pos):
        posz = fmdj.ztree.distributed_zsort(pos, fmdj.config.CommunicationConfig())
        return posz

    return lambda pos=1: f(pos) # Hack to make function accept kw arguments

def get_mesh(ndev=-1):
    return jax.sharding.Mesh(jax.devices()[:ndev], ('gpus',), axis_types=(AxisType.Auto))

@pytest.mark.parametrize("ndev", NDEVS)
@pytest.mark.multi_gpu
def bench_multi_zsort(jax_bench, ndev):
    Ntot = 1024*1024*32
    
    pos = jax.jit(mk_pos(Ntot, alloc_fac=1.2, ndev=ndev))()
    fzs = jax.jit(mksort(ndev=ndev))

    jb = jax_bench(jit_rounds=20, jit_warmup=2, eager_rounds=0, eager_warmup=0)

    posz = jb.measure(fn_jit=fzs, pos=pos, tag="random")[1]
    posz1 = jb.measure(fn_jit=fzs, pos=posz, tag="sorted")[1]
    posz2 = jb.measure(fn_jit=fzs, pos=posz + 1e-2*pos, tag="displaced")[1]