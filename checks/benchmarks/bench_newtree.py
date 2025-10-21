import fmdj
import pytest
from fmdj.config import Config
import jax
import custom_jax as cj
import jax.numpy as jnp
import fmdj.new_tree as nt


@pytest.fixture
def particlesz(request):
    npart = request.param if hasattr(request, "param") else 1024*1024
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    posz, isort = cj.tree.pos_zorder_sort(pos0)
    return nt.Particles(posz, jnp.ones(posz.shape[0]))

@pytest.mark.parametrize("particlesz", [1024*128,1024*1024, 1024*1024*8], indirect=True)
def bench_tree_hierarchy(jax_bench, particlesz):
    jb = jax_bench(jit_rounds=50, jit_warmup=5, eager_rounds=3, eager_warmup=1)
    
    cfg = Config(tags=("cuda", "base"), tree=nt.TreeConfig(coarse_fac=8.0))

    jb.measure(
        fn=nt.build_tree_hierarchy, fn_jit=nt.build_tree_hierarchy.jit, 
        part=particlesz,
        cfg=cfg)