import pytest
from fmdj.config import FMMConfig
from dataclasses import replace
from jztree.tree import build_tree_hierarchy, zsort, center_of_mass
from fmdj.multipoles import build_multipole_hierarchy
from fmdj.fmm import _fmm_dual_walk, leaf_leaf_summation, fast_multipole_method
from fmdj.multipoles import _fmm_node_to_child, summarize_multipoles
from fmdj.data import PosMass
import jax
from jztree_utils import ics

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("coarsen_fac", [2,4,6,8])
def bench_n2n_coarsen(jax_bench, pos_mass_z, coarsen_fac):
    cfg_fmm = FMMConfig()
    cfg_fmm = replace(cfg_fmm, tree=replace(cfg_fmm.tree, coarse_fac=coarsen_fac))
    if coarsen_fac <= 4:
        cfg_fmm = replace(cfg_fmm, tree=replace(cfg_fmm.tree, alloc_fac_nodes=1.5))
    
    th = build_tree_hierarchy.jit(pos_mass_z, cfg_fmm.tree)
    mph = build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg_fmm=cfg_fmm)
    
    jb = jax_bench(jit_rounds=100, jit_warmup=50)

    jb.measure(fn_jit=_fmm_dual_walk.jit,
        th=th, mph=mph, cfg_fmm=cfg_fmm
    )

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("max_leaf_size", [16,24,32,48])
def bench_leaf_size(jax_bench, pos_mass_z, max_leaf_size):
    cfg_fmm = FMMConfig()
    cfg_fmm = replace(cfg_fmm, tree=replace(cfg_fmm.tree, max_leaf_size=max_leaf_size))

    jb = jax_bench(jit_rounds=40, jit_warmup=20)

    th = build_tree_hierarchy.jit(pos_mass_z, cfg_fmm.tree)
    mph = build_multipole_hierarchy.jit(th, pos_mass_z.pos, pos_mass_z.mass, cfg_fmm=cfg_fmm)

    res, (loc, ilist) = jb.measure(fn_jit=_fmm_dual_walk.jit,
        th=th, mph=mph, cfg_fmm=cfg_fmm, tag="node2node"
    )

    jb.measure(fn_jit=leaf_leaf_summation.jit,
        particles=pos_mass_z, ispl=th.splits_leaf_to_part(), ilist=ilist, cfg_fmm=cfg_fmm,
        tag="leaf2leaf"
    )

# @pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("npart", [1024*128, 1024*1024, 1024*1024*4, 8*1024*1024])
def bench_fmm_npart(jax_bench, pos_mass_z):
    cfg_fmm = FMMConfig()
    jb = jax_bench(jit_rounds=20, jit_warmup=2)

    jb.measure(fn_jit=fast_multipole_method.jit,
               part=pos_mass_z, cfg_fmm=cfg_fmm)

@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("p", [1,2,3,4,5])
def bench_fmm_p(jax_bench, p, pos_mass_z):
    cfg_fmm = FMMConfig(p=p)

    jb = jax_bench(jit_rounds=20, jit_warmup=2)
    jb.measure(fn_jit=fast_multipole_method.jit,
               part=pos_mass_z, cfg_fmm=cfg_fmm)

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("p", [3,4,5,6])
def bench_fmm_steps(jax_bench, p):
    cfg_fmm = FMMConfig(p=p)

    # jb = jax_bench(jit_rounds=40, jit_warmup=10) # for more timing accuracy
    jb = jax_bench(jit_rounds=12, jit_warmup=1) 

    pos_mass = ics.uniform_particles(int(1e6), total_mass=1.0, seed=0)
    # pos_mass = ics.uniform_particles(int(1e7), total_mass=1.0, seed=0) # for comparing against multi-GPU

    pos_mass_z = jb.measure(fn_jit=zsort.jit, pos=pos_mass, tag="zsort")[1][0]

    th = jb.measure(fn_jit=build_tree_hierarchy.jit, partz=pos_mass_z, cfg_tree=cfg_fmm.tree, tag="build_new")[1]

    mph = jb.measure(fn_jit=build_multipole_hierarchy.jit, 
                     th=th, pos=pos_mass_z.pos, mp=pos_mass_z.mass, cfg_fmm=cfg_fmm, tag="multipoles")[1]

    loc, ilist = jb.measure(fn_jit=_fmm_dual_walk.jit, 
                            th=th, mph=mph, cfg_fmm=cfg_fmm, tag="node2node")[1]
    phif = jb.measure(fn_jit=_fmm_node_to_child.jit, 
                      ispl=th.splits_leaf_to_part(), loc=loc, xnode=th.center().get(0, th.size()), xchild=pos_mass_z.pos,
                      cfg_fmm=cfg_fmm, pout=1,tag="loc2loc")[1]
    fphi = jb.measure(fn_jit=leaf_leaf_summation.jit,
                      particles=pos_mass_z, ispl=th.splits_leaf_to_part(), ilist=ilist, cfg_fmm=cfg_fmm, tag="leaf2leaf")[1]
    
    jb.measure(fn_jit=fast_multipole_method.jit,
        part=pos_mass, cfg_fmm=cfg_fmm, result="loc", tag=f"total"
    )

    jb.measure(fn_jit=fast_multipole_method.jit,
        part=pos_mass, cfg_fmm=cfg_fmm, result="partz_locz", tag=f"totalrz"
    )

@pytest.mark.shrink_in_quick(keep_index=0)
@pytest.mark.parametrize("p", [3,4,5])
def bench_particle_multipoles(jax_bench, p, pos_mass_z, tree_hierarchy):
    th = tree_hierarchy
    ispl = th.splits_leaf_to_part()
    cent =  th.center().get(0, th.size())

    jb = jax_bench(jit_rounds=200, jit_warmup=20)

    cfg_fmm = FMMConfig(p=p)

    jb.measure(
        fn_jit = summarize_multipoles.jit,
        ispl=ispl, mp=pos_mass_z.mass, xnode=cent, xchild=pos_mass_z.pos, cfg_fmm=cfg_fmm,
        tag="part2mp"
    )
    
    jb.measure(fn_jit = center_of_mass.jit,
               ispl=ispl, part=pos_mass_z,
               tag="com"
    )
