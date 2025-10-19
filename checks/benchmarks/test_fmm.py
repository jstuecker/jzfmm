import fmdj
import pytest

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

@pytest.fixture
def interactions(particles):
    pos0, mass0 = particles
    octree, posz, massz, isortz = fmdj.fmm.build_octree_with_multipoles.jit(pos0+0.5, mass0, p=2)
    err, (ilist, nilist) = fmdj.fmm.build_interaction_list.jit(octree)
    err.throw()
    ilist, iranges = fmdj.fmm.organize_interactions.jit(ilist, nilist, sort=False)
    ilist.block_until_ready()
    return octree, posz, massz, ilist, nilist, iranges

@pytest.mark.parametrize("particles", [1024*256,1024*1024, 1024*1024*2], indirect=True)
def test_interactions(jax_bench, interactions):
    octree, posz, massz, ilist, nilist, iranges = interactions

    jb = jax_bench(jit_rounds=10, jit_warmup=1, eager_rounds=0, eager_warmup=0)

    for mode in "base", "cuda":
        cfg = fmdj.Config(tags=(mode, "base"), p=octree.p, softening=1e-3)

        Loc = jb.measure(
            fmdj.multipoles.ilist_node_to_node, fmdj.multipoles.ilist_node_to_node.jit,
            octree.xnode, octree.mp, ilist, iranges, cfg=cfg,
            tag=f"n2n-{mode}"
        )[1]

        jb.measure(
            fmdj.multipoles.ilist_leaf_to_node, fmdj.multipoles.ilist_leaf_to_node.jit,
            octree.xnode, posz, massz, octree.leaf_particle_bounds, ilist, iranges, cfg=cfg,
            tag=f"l2n-{mode}"
        )

        jb.measure(
            fmdj.multipoles.ilist_leaf_to_leaf, fmdj.multipoles.ilist_leaf_to_leaf.jit,
            posz, massz, octree.leaf_particle_bounds, ilist, iranges, cfg=cfg,
            tag=f"l2l-{mode}"
        )

        if mode == "base":
            jb.measure(
                fmdj.multipoles.ilist_node_to_leaf, fmdj.multipoles.ilist_node_to_leaf.jit,
                octree.xnode, octree.mp, posz, octree.leaf_particle_bounds, ilist, iranges, cfg=cfg,
                tag="n2l-base"
            )

            Loc = jb.measure(
                fmdj.multipoles.local_to_local_via_height, fmdj.multipoles.local_to_local_via_height.jit,
                octree, Loc,
                tag="Loc-Down"
            )[1]
            phi = jb.measure(
                fmdj.multipoles.evaluate_local, fmdj.multipoles.evaluate_local.jit,
                Loc[octree.node_of_particle], posz - octree.xnode[octree.node_of_particle],
                tag="Loc-Eval"
            )[1]

        jb.measure(
            fmdj.fmm.evaluate_interaction_lists, fmdj.fmm.evaluate_interaction_lists.jit,
            octree, posz, massz, ilist, nilist, sort=False, cfg=cfg,
            tag=f"tot-{mode}"
        )