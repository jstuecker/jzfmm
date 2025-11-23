import fmdj
import pytest
from fmdj.config import Config, FMMConfig, LoggingConfig
import jax
import fmdj_ffi as cj
import jax.numpy as jnp
import fmdj.new_tree as nt
import fmdj_ffi.cj_new_tree as cnt


@pytest.fixture
def particlesz(request):
    npart = request.param if hasattr(request, "param") else 1024*1024
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    posz, isort = cj.tree.pos_zorder_sort(pos0)
    return nt.Particles(posz, jnp.ones(posz.shape[0]))

@pytest.mark.parametrize("particlesz", [1024*128,1024*1024, 1024*1024*8], indirect=True)
def bench_tree_hierarchy(jax_bench, particlesz):
    jb = jax_bench(jit_rounds=50, jit_warmup=5, eager_rounds=3, eager_warmup=1)
    
    cfg = Config(tags=("cuda", "base"), fmm=nt.FMMConfig(coarse_fac=8.0))

    jb.measure(
        fn=nt.build_tree_hierarchy, fn_jit=nt.build_tree_hierarchy.jit, 
        part=particlesz,
        cfg=cfg)
    
def bench_cuda(jax_bench, particlesz):
    cfg = Config(fmm=FMMConfig(alloc_fac_nodes=1.2, coarse_fac=4.0, p=2, stop_coarsen=512, ilist_alloc_fac=2048),
                logging=LoggingConfig(level=0))

    th = nt.build_tree_hierarchy(particlesz, cfg)
    loc, ilist = nt.evaluate_plane_interactions(th[-1], cfg=cfg)
    th = nt.build_tree_hierarchy(particlesz, cfg)

    cfg.tags = ("base",)
    loc, ilist = nt.evaluate_plane_interactions(th[-1], cfg=cfg)
    loc2, ilist2 = nt.evaluate_plane_interactions.jit(th[-2], th[-1], ilist, loc, cfg=cfg)
    loc3, ilist3 = nt.evaluate_plane_interactions.jit(th[-3], th[-2], ilist2, loc2, cfg=cfg)
    loc4, ilist4 = nt.evaluate_plane_interactions.jit(th[-4], th[-3], ilist3, loc3, cfg=cfg)
    loc5, ilist5 = nt.evaluate_plane_interactions.jit(th[-5], th[-4], ilist4, loc4*0., cfg=cfg)

    jax.block_until_ready((loc4, ilist4, th, loc5, ilist5))

    cfg.tags = ("cuda", "base")
    jb = jax_bench(jit_rounds=200, jit_warmup=100, eager_rounds=0, eager_warmup=0)
    # jb.measure(
    #     plane=th[-5], plane_lr=th[-4], ilist_lr=ilist4, loc_lr=loc4, cfg=cfg,
    #     fn_jit=nt.evaluate_plane_interactions.jit, tag="jax"
    # )

    bdata, (lnew, inew) = jb.measure(
        plane=th[-5], plane_lr=th[-4], ilist_lr=ilist4, loc_lr=loc4*0., cfg=cfg,
        fn_jit=cnt.cj_evaluate_tree_plane.jit,
    )

    nnodes = th[-5].nnodes
    n5 = ilist5.ispl[1:] - ilist5.ispl[:-1]
    nnew = inew.ispl[1:] - inew.ispl[:-1]
    
    # There are some edge cases where the opening criterion triggers differently
    # I assume that these cases are triggered by roundoff errors.
    # This leads to legit differnces in the local terms, so we mask them out
    mask = n5[:nnodes] == nnew[:nnodes]
    print("differently opened:", jnp.where(~mask))
    assert jnp.sum(~mask) < 100

    assert lnew[:nnodes][mask] == pytest.approx(loc5[:nnodes][mask], rel=1e-3, abs=1e-1)

@pytest.fixture
def leaf_leaf_ilist(particlesz, request):
    cfg = Config(fmm=FMMConfig(alloc_fac_nodes=1.2, coarse_fac=4.0, p=2),
                logging=LoggingConfig(level=0))
    cfg.fmm.opening_angle = 1.0
    cfg.fmm.max_leaf_size = request.param if hasattr(request, "param") else 32
    cfg.softening = 0.1

    th = nt.build_tree_hierarchy.jit(particlesz, cfg)
    loc, ilist = nt.evaluate_interaction_hierarchy.jit(th, cfg=cfg)
    return particlesz, th[0], ilist, cfg

@pytest.mark.parametrize("leaf_leaf_ilist", [12,16,24,32], indirect=True)
def bench_leaf_leaf(jax_bench, leaf_leaf_ilist):
    particlesz, plane, ilist, cfg = leaf_leaf_ilist

    jb = jax_bench(jit_rounds=20, jit_warmup=3)

    irange = jnp.array([0, ilist.nfilled], dtype=jnp.int32)
    i0 = nt.inverse_of_splits(ilist.ispl, ilist.size())
    il = jnp.stack([i0, ilist.iother], axis=1)

    res, phi = jb.measure(
        xpart=particlesz.pos, mpart=particlesz.mass, leaf_bounds=plane.ispl, interactions=il, 
        irange=irange, cfg=cfg,
        fn_jit=fmdj.multipoles.ilist_leaf_to_leaf.jit,
        tag="cu_leaf2leaf"
    )

    res, fphi = jb.measure(particles=particlesz, plane=plane, ilist=ilist, cfg=cfg,
        fn_jit=cnt.cj_new_force_and_pot.jit,
        tag="new_leaf2leaf"
    )

    assert fphi[:,3] == pytest.approx(phi, rel=1e-2, abs=20.)

def measure_fmm(jb, cfg, npart=1024**2):
    pos0 = jax.random.normal(jax.random.PRNGKey(0), (npart,3))
    mass = jnp.ones_like(pos0[:,0])

    res, phi2 = jb.measure(fn_jit=fmdj.fmm.new_fmm_potential.jit, tag="new_fmm",
               pos=pos0, mass=mass, cfg=cfg)

    if npart <= 5e6:
        res, phi1 = jb.measure(fn_jit=fmdj.fmm.fast_multipole_potential.jit, tag="old_fmm",
                pos=pos0, mass=mass, cfg=cfg)
    
        assert phi1 == pytest.approx(phi2, rel=1e-1, abs=1.0)

@pytest.mark.parametrize("npart", [1024*128, 1024*1024, 1024*1024*4, 8*1024*1024])
def bench_fmm_npart(jax_bench, npart):
    cfg = Config(fmm=FMMConfig(p=2, max_leaf_size=32), softening=1e-2)
    cfg.fmm.opening_angle = 1.0

    jb = jax_bench(jit_rounds=20, jit_warmup=2)
    
    measure_fmm(jb, cfg, npart=npart)

@pytest.mark.parametrize("p", [1,2,3,4,5])
def bench_fmm_p(jax_bench, p):
    cfg = Config(fmm=FMMConfig(p=p, max_leaf_size=32), softening=1e-2)
    cfg.fmm.opening_angle = 1.0

    jb = jax_bench(jit_rounds=20, jit_warmup=3)
    
    measure_fmm(jb, cfg)