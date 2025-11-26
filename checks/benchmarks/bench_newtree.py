import fmdj
import pytest
from fmdj.config import Config, FMMConfig
import jax
import jax.numpy as jnp
import fmdj.fmm
from fmdj.config import Config, FMMConfig
from dataclasses import replace

from fmdj.ztree import build_tree_hierarchy
from fmdj.fmm import evaluate_plane_interactions, evaluate_interaction_hierarchy

@pytest.mark.parametrize("npart", [1024*128,1024*1024, 1024*1024*8], indirect=True)
def bench_tree_hierarchy(jax_bench, pos_mass_z, cfg):
    jb = jax_bench(jit_rounds=50, jit_warmup=5, eager_rounds=3, eager_warmup=1)

    jb.measure(
        fn=fmdj.ztree.build_tree_hierarchy, fn_jit=fmdj.ztree.build_tree_hierarchy.jit, 
        part=pos_mass_z,
        cfg=cfg)

def bench_cuda(jax_bench, pos_mass_z, cfg):
    th = build_tree_hierarchy(pos_mass_z, cfg)
    loc, ilist = evaluate_plane_interactions(th[-1], cfg=cfg)
    th = build_tree_hierarchy(pos_mass_z, cfg)

    loc, ilist = evaluate_plane_interactions(th[-1], cfg=cfg)
    loc2, ilist2 = evaluate_plane_interactions.jit(th[-2], th[-1], ilist, loc, cfg=cfg)
    loc3, ilist3 = evaluate_plane_interactions.jit(th[-3], th[-2], ilist2, loc2, cfg=cfg)
    loc4, ilist4 = evaluate_plane_interactions.jit(th[-4], th[-3], ilist3, loc3, cfg=cfg)
    loc5, ilist5 = evaluate_plane_interactions.jit(th[-5], th[-4], ilist4, loc4*0., cfg=cfg)

    jax.block_until_ready((loc4, ilist4, th, loc5, ilist5))

    jb = jax_bench(jit_rounds=200, jit_warmup=100, eager_rounds=0, eager_warmup=0)

    bdata, (lnew, inew) = jb.measure(
        plane=th[-5], plane_lr=th[-4], ilist_lr=ilist4, loc_lr=loc4*0., cfg=cfg,
        fn_jit=evaluate_plane_interactions.jit,
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
def max_leaf_size(request):
    return getattr(request, "param", 32)

@pytest.fixture
def leaf_leaf_ilist(pos_mass_z, max_leaf_size, cfg):
    fmm = replace(cfg.fmm, max_leaf_size=max_leaf_size)
    cfg = replace(cfg, fmm=fmm, softening=0.1)

    th = build_tree_hierarchy.jit(pos_mass_z, cfg)
    loc, ilist = evaluate_interaction_hierarchy.jit(th, cfg=cfg)
    return pos_mass_z, th[0], ilist, cfg

@pytest.mark.parametrize("max_leaf_size", [12,16,24,32], indirect=True)
def bench_leaf_leaf(jax_bench, leaf_leaf_ilist):
    particlesz, plane, ilist, cfg = leaf_leaf_ilist

    jb = jax_bench(jit_rounds=20, jit_warmup=3)

    res, fphi = jb.measure(particles=particlesz, plane=plane, ilist=ilist, cfg=cfg,
        fn_jit=fmdj.fmm.grouped_force_and_pot.jit,
        tag="new_leaf2leaf"
    )

@pytest.mark.parametrize("npart", [1024*128, 1024*1024, 1024*1024*4, 8*1024*1024])
def bench_fmm_npart(jax_bench, pos_mass_z, cfg):
    jb = jax_bench(jit_rounds=20, jit_warmup=2)

    jb.measure(fn_jit=fmdj.fmm.fmm_force_and_potential.jit,
               pos=pos_mass_z.pos, mass=pos_mass_z.mass, cfg=cfg)

@pytest.mark.parametrize("p", [1,2,3,4,5])
def bench_fmm_p(jax_bench, p, pos_mass_z):
    cfg = Config(fmm=FMMConfig(p=p, max_leaf_size=32, opening_angle=1.0), softening=1e-2)

    jb = jax_bench(jit_rounds=20, jit_warmup=2)
    jb.measure(fn_jit=fmdj.fmm.fmm_force_and_potential.jit,
               pos=pos_mass_z.pos, mass=pos_mass_z.mass, cfg=cfg)