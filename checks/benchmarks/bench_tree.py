import fmdj
import pytest

@pytest.mark.parametrize("npart", [1024*128,1024*1024, 1024*1024*8], indirect=True)
def bench_build_tree_hierarchy(jax_bench, pos_mass_z, cfg):
    jb = jax_bench(jit_rounds=50, jit_warmup=5, eager_rounds=3, eager_warmup=1)

    jb.measure(
        fn=fmdj.ztree.build_tree_hierarchy, fn_jit=fmdj.ztree.build_tree_hierarchy.jit, 
        part=pos_mass_z, cfg_tree=cfg.tree, tag="new")
    
@pytest.mark.parametrize("npart", [1024*128,1024*1024,1024*1024*8], indirect=True)
def bench_zsort(jax_bench, pos_mass, cfg):
    jb = jax_bench(jit_rounds=50, jit_warmup=5)

    jb.measure(fn_jit=fmdj.ztree.pos_zorder_sort.jit, x=pos_mass.pos)