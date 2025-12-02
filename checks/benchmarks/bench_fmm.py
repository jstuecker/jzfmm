import fmdj
import pytest
from fmdj.config import Config, FMMConfig
import jax
import jax.numpy as jnp
import fmdj.fmm
from fmdj.config import Config, FMMConfig
from dataclasses import replace

@pytest.mark.parametrize("coarsen_fac", [2,4,6,8])
def bench_n2n_coarsen(jax_bench, pos_mass_z, cfg, coarsen_fac):
    cfg = replace(cfg, fmm=replace(cfg.fmm, coarse_fac=coarsen_fac))
    th = fmdj.ztree.build_tree_hierarchy.jit(pos_mass_z, cfg)
    mph = fmdj.multipoles.build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg=cfg)
    
    jb = jax_bench(jit_rounds=100, jit_warmup=50)

    jb.measure(fn_jit=fmdj.fmm.evaluate_interaction_hierarchy.jit,
        th=th, mph=mph, cfg=cfg
    )

@pytest.mark.parametrize("max_leaf_size", [16,24,32,48])
def bench_leaf_size(jax_bench, pos_mass_z, cfg, max_leaf_size):
    cfg = replace(cfg, fmm=replace(cfg.fmm, max_leaf_size=max_leaf_size))

    jb = jax_bench(jit_rounds=40, jit_warmup=20)

    th = fmdj.ztree.build_tree_hierarchy.jit(pos_mass_z, cfg)
    mph = fmdj.multipoles.build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg=cfg)

    res, (loc, ilist) = jb.measure(fn_jit=fmdj.fmm.evaluate_interaction_hierarchy.jit,
        th=th, mph=mph, cfg=cfg, tag="node2node"
    )

    jb.measure(fn_jit=fmdj.fmm.grouped_force_and_pot.jit,
        particles=pos_mass_z, ispl=th[0].ispl, ilist=ilist, cfg=cfg,
        tag="leaf2leaf"
    )

@pytest.mark.parametrize("npart", [1024*128, 1024*1024, 1024*1024*4, 8*1024*1024])
def bench_fmm_npart(jax_bench, pos_mass_z, cfg):
    jb = jax_bench(jit_rounds=20, jit_warmup=2)

    jb.measure(fn_jit=fmdj.fmm.fast_multipole_method.jit,
               part=pos_mass_z, cfg=cfg)

@pytest.mark.parametrize("p", [1,2,3,4,5])
def bench_fmm_p(jax_bench, p, pos_mass_z):
    cfg = Config(fmm=FMMConfig(p=p))

    jb = jax_bench(jit_rounds=20, jit_warmup=2)
    jb.measure(fn_jit=fmdj.fmm.fast_multipole_method.jit,
               part=pos_mass_z, cfg=cfg)

@pytest.mark.parametrize("p", [3,4,5])
def bench_fmm_steps(jax_bench, p, pos_mass):
    cfg = Config(fmm=FMMConfig(p=p))

    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    posz, isortz = jb.measure(fn_jit=fmdj.ztree.pos_zorder_sort.jit, x=pos_mass.pos, tag="zsort")[1]
    pos_mass_z = fmdj.data.PosMass(pos=posz, mass=pos_mass.mass[isortz])

    th = jb.measure(fn_jit=fmdj.ztree.build_tree_hierarchy.jit, part=pos_mass_z, cfg=cfg, tag="build")[1]
    mph = jb.measure(fn_jit=fmdj.multipoles.build_multipole_hierarchy.jit, 
                     th=th, pos=pos_mass_z.pos, mp=pos_mass_z.mass, cfg=cfg, tag="multipoles")[1]

    loc, ilist = jb.measure(fn_jit=fmdj.fmm.evaluate_interaction_hierarchy.jit, 
                            th=th, mph=mph, cfg=cfg, tag="node2node")[1]
    parent = th[0].icoarse_of_fine()
    phif = jb.measure(fn_jit=fmdj.fmm.shift_local_to_children.jit, 
                      ispl=th[0].ispl, loc=loc, xnode=th[0].center(), xchild=pos_mass_z.pos,
                      cfg=cfg, pout=1,tag="loc2loc")[1]
    fphi = jb.measure(fn_jit=fmdj.fmm.grouped_force_and_pot.jit,
                      particles=pos_mass_z, ispl=th[0].ispl, ilist=ilist, cfg=cfg, tag="leaf2leaf")[1]

@pytest.mark.parametrize("p", [3,4,5])
def bench_particle_multipoles(jax_bench, p, pos_mass_z, tree_hierarchy):
    th = tree_hierarchy

    jb = jax_bench(jit_rounds=200, jit_warmup=20)

    cfg = Config(fmm=FMMConfig(p=p, multipoles_around_com=False))

    jb.measure(
        fn_jit = fmdj.multipoles.summarize_multipoles.jit,
        ispl=th[0].ispl, mp=pos_mass_z.mass, xnode=th[0].center(), xchild=pos_mass_z.pos, cfg=cfg,
        tag="part2mp"
    )
    
    jb.measure(fn_jit = fmdj.multipoles.center_of_mass.jit,
               ispl=th[0].ispl, part=pos_mass_z, cfg=cfg,
               tag="com"
    )