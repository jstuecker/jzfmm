import pytest
from fmdj.config import Config, FMMConfig
import jax
import jax.numpy as jnp
from fmdj.config import Config, FMMConfig
from dataclasses import replace
from fmdj.data import PosMass
from jztree.tree import build_tree_hierarchy, pos_zorder_sort, center_of_mass
from fmdj.multipoles import build_multipole_hierarchy
from fmdj.fmm import evaluate_interaction_hierarchy, grouped_force_and_pot, fast_multipole_method
from fmdj.multipoles import shift_local_to_children, summarize_multipoles

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("coarsen_fac", [2,4,6,8])
def bench_n2n_coarsen(jax_bench, pos_mass_z, cfg, coarsen_fac):
    cfg = replace(cfg, tree=replace(cfg.tree, coarse_fac=coarsen_fac))
    if coarsen_fac <= 4:
        cfg = replace(cfg, tree=replace(cfg.tree, alloc_fac_nodes=1.5))
    
    th = build_tree_hierarchy.jit(pos_mass_z, cfg.tree)
    mph = build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg=cfg)
    
    jb = jax_bench(jit_rounds=100, jit_warmup=50)

    jb.measure(fn_jit=evaluate_interaction_hierarchy.jit,
        th=th, mph=mph, cfg=cfg
    )

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("max_leaf_size", [16,24,32,48])
def bench_leaf_size(jax_bench, pos_mass_z, cfg, max_leaf_size):
    cfg = replace(cfg, tree=replace(cfg.tree, max_leaf_size=max_leaf_size))

    jb = jax_bench(jit_rounds=40, jit_warmup=20)

    th = build_tree_hierarchy.jit(pos_mass_z, cfg.tree)
    mph = build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg=cfg)

    res, (loc, ilist) = jb.measure(fn_jit=evaluate_interaction_hierarchy.jit,
        th=th, mph=mph, cfg=cfg, tag="node2node"
    )

    jb.measure(fn_jit=grouped_force_and_pot.jit,
        particles=pos_mass_z, ispl=th.ispl_n2n.get(0, th.base_size()), ilist=ilist, cfg=cfg,
        tag="leaf2leaf"
    )

# @pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [1024*128, 1024*1024, 1024*1024*4, 8*1024*1024])
def bench_fmm_npart(jax_bench, pos_mass_z, cfg):
    jb = jax_bench(jit_rounds=20, jit_warmup=2)

    jb.measure(fn_jit=fast_multipole_method.jit,
               part=pos_mass_z, cfg=cfg)

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("p", [1,2,3,4,5])
def bench_fmm_p(jax_bench, p, pos_mass_z):
    cfg = Config(fmm=FMMConfig(p=p))

    jb = jax_bench(jit_rounds=20, jit_warmup=2)
    jb.measure(fn_jit=fast_multipole_method.jit,
               part=pos_mass_z, cfg=cfg)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("p", [3,4,5])
def bench_fmm_steps(jax_bench, p, pos_mass):
    cfg = Config(fmm=FMMConfig(p=p))

    jb = jax_bench(jit_rounds=40, jit_warmup=10)

    posz, isortz = jb.measure(fn_jit=pos_zorder_sort.jit, x=pos_mass.pos, tag="zsort")[1]
    pos_mass_z = PosMass(pos=posz, mass=pos_mass.mass[isortz])

    th = jb.measure(fn_jit=build_tree_hierarchy.jit, part=pos_mass_z, cfg_tree=cfg.tree, tag="build_new")[1]

    mph = jb.measure(fn_jit=build_multipole_hierarchy.jit, 
                     th=th, pos=pos_mass_z.pos, mp=pos_mass_z.mass, cfg=cfg, tag="multipoles")[1]

    loc, ilist = jb.measure(fn_jit=evaluate_interaction_hierarchy.jit, 
                            th=th, mph=mph, cfg=cfg, tag="node2node")[1]
    phif = jb.measure(fn_jit=shift_local_to_children.jit, 
                      ispl=th.ispl_n2n.get(0, th.base_size()+1), loc=loc, xnode=th.center().get(0, th.base_size()), xchild=pos_mass_z.pos,
                      cfg=cfg, pout=1,tag="loc2loc")[1]
    fphi = jb.measure(fn_jit=grouped_force_and_pot.jit,
                      particles=pos_mass_z, ispl=th.ispl_n2n.get(0, th.base_size()), ilist=ilist, cfg=cfg, tag="leaf2leaf")[1]

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("p", [3,4,5])
def bench_particle_multipoles(jax_bench, p, pos_mass_z, tree_hierarchy):
    th = tree_hierarchy
    ispl = th.ispl_n2n.get(0, th.base_size()+1)
    cent =  th.center().get(0, th.base_size())

    jb = jax_bench(jit_rounds=200, jit_warmup=20)

    cfg = Config(fmm=FMMConfig(p=p))

    jb.measure(
        fn_jit = summarize_multipoles.jit,
        ispl=ispl, mp=pos_mass_z.mass, xnode=cent, xchild=pos_mass_z.pos, cfg=cfg,
        tag="part2mp"
    )
    
    jb.measure(fn_jit = center_of_mass.jit,
               ispl=ispl, part=pos_mass_z,
               tag="com"
    )