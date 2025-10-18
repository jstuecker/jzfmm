import fmdj

def test_fmm_steps(jax_bench, particles):
    jb = jax_bench(jit_rounds=10, jit_warmup=1)
    
    octree, pos, mass, isort = jb.measure(
        fn_jit=fmdj.fmm.build_octree_with_multipoles.jit,
        pos=particles[0], mass=particles[1], max_leaf_size=64, p=3,
        tag = "octree"
    )[1]

    err, (ilist, nilist) = jb.measure(
        fn_jit=fmdj.fmm.build_interaction_list.jit,
        octree=octree, thetamax=0.75, ilist_fac=512,
        tag = "ilist"
    )[1]
    err.throw()

    ilist, isplits = jb.measure(
        fn_jit=fmdj.fmm.organize_interactions.jit, 
        interaction_list=ilist, nfilled=nilist, sort=False,
        tag="organize"
    )[1]