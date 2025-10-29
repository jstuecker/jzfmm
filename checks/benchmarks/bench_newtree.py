import fmdj
import pytest
from fmdj.config import Config, TreeConfig, LoggingConfig
import jax
import custom_jax as cj
import jax.numpy as jnp
import fmdj.new_tree as nt
import custom_jax.cj_new_tree as cnt


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
    
def bench_cuda(jax_bench, particlesz):
    print("Starting Test!")
    cfg = Config(p=2, tree=TreeConfig(alloc_fac_nodes=1.2, coarse_fac=4.0, p=2, stop_coarsen=512, ilist_alloc_fac=2048),
                logging=LoggingConfig(level=0))

    th = nt.build_tree_hierarchy(particlesz, cfg)
    loc, ilist = nt.evaluate_plane_interactions(th[-1], cfg=cfg)
    th = nt.build_tree_hierarchy(particlesz, cfg)

    loc, ilist = nt.evaluate_plane_interactions(th[-1], cfg=cfg)
    loc2, ilist2 = nt.evaluate_plane_interactions.jit(th[-2], th[-1], ilist, loc, cfg=cfg)
    loc3, ilist3 = nt.evaluate_plane_interactions.jit(th[-3], th[-2], ilist2, loc2, cfg=cfg)
    loc4, ilist4 = nt.evaluate_plane_interactions.jit(th[-4], th[-3], ilist3, loc3, cfg=cfg)
    # loc5, ilist5 = nt.evaluate_plane_interactions.jit(th[-5], th[-4], ilist4, loc4, cfg=cfg)

    jb = jax_bench(jit_rounds=20, jit_warmup=5, eager_rounds=0, eager_warmup=0)
    jb.measure(
        plane=th[-5], plane_lr=th[-4], ilist_lr=ilist4, loc_lr=loc4, cfg=cfg,
        fn_jit=nt.evaluate_plane_interactions.jit, tag="jax"
    )

    bdata, (lnew, inew) = jb.measure(
        plane=th[-5], plane_lr=th[-4], ilist_lr=ilist4, loc_lr=loc4, cfg=cfg,
        fn_jit=cnt.cj_evaluate_tree_plane.jit, tag="cuda_mpread"
    )
    print("Ended!!")

    print(lnew)
    print(inew.ispl)