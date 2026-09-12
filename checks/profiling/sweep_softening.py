"""Time softened-opening tree walk plus near field on one fixed uniform tree."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from jztree.data import PosLvl
from jztree_utils import ics
from jzfmm.config import FMMConfig, OpeningBySoftenedAngle, PlummerKernel
from jzfmm.fmm import fast_multipole_method, _fmm_dual_walk, _leaf_leaf_summation
from jzfmm.multipoles import build_multipole_hierarchy, _fmm_node_to_child


def walk_near(partz, tree, mph, cfg):
    loc, ilist = _fmm_dual_walk(tree, mph, cfg_fmm=cfg)
    spl = tree.splits_leaf_to_part()
    loc_part = _fmm_node_to_child(spl, loc, tree.poslvl(0),
        PosLvl(pos=partz.pos, lvl=jnp.zeros(partz.pos.shape[0], dtype=jnp.int32)),
        pout=1, cfg_fmm=cfg)
    return _leaf_leaf_summation(partz, spl, ilist, cfg_fmm=cfg, loc_in=loc_part)


walk_jit = jax.jit(walk_near, static_argnames=['cfg'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,default=4_000_000)
    parser.add_argument('--points',type=int,default=19)
    parser.add_argument('--rounds',type=int,default=30)
    parser.add_argument('--trace-rounds',type=int,default=10)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='gpu'
    part=jax.block_until_ready(ics.uniform_particles(args.n,seed=0))
    base=FMMConfig()
    partz,tree=jax.block_until_ready(fast_multipole_method.jit(part,base,result='partz_tree'))
    mph=jax.block_until_ready(jax.jit(build_multipole_hierarchy,static_argnames=['cfg_fmm'])(
        tree,partz.pos,jnp.reshape(partz.mass,jnp.shape(partz.mass)+(1,)),cfg_fmm=base))
    ell=args.n**(-1/3)
    ratios=np.logspace(-1,2,args.points)
    # Fixed randomized order to avoid making clock/thermal drift track epsilon.
    order=np.random.default_rng(42).permutation(args.points)
    metadata=dict(n=args.n,seed=0,spacing=ell,ratios=ratios.tolist(),execution_order=order.tolist(),
        total_definition='Prebuilt tree and multipoles: dual walk, local-to-local shifts, local-to-particle translation, leaf-leaf; synchronized wall time',
        nodes_per_plane=[int(tree.num(p)) for p in range(tree.num_planes())])
    (args.output/'sweep.json').write_text(json.dumps(metadata,indent=2))
    for i in order:
        ratio=float(ratios[i]);eps=ratio*ell
        dest=args.output/f'point_{i:02d}';dest.mkdir(exist_ok=True)
        cfg=FMMConfig(kernel=PlummerKernel(softening=eps),opening=OpeningBySoftenedAngle(softening=eps))
        call=lambda:jax.block_until_ready(walk_jit(partz,tree,mph,cfg))
        print(f'Start {i}: epsilon/ell={ratio:.6g}',flush=True)
        for _ in range(4):out=call()
        samples=[]
        for _ in range(args.rounds):
            t=time.perf_counter();out=call();samples.append((time.perf_counter()-t)*1000)
        finite=bool(np.isfinite(np.asarray(out)).all());assert finite
        # Verify the isolated pipeline against the normal existing-tree evaluator.
        comparison=None
        if i in [0,args.points//2,args.points-1]:
            reference=jax.block_until_ready(fast_multipole_method.jit(partz,cfg,th=tree,result='locz'))
            difference=np.max(np.abs(np.asarray(out)-np.asarray(reference.values)))
            np.testing.assert_allclose(np.asarray(out),np.asarray(reference.values),rtol=2e-6,atol=2e-6)
            comparison=float(difference)
        timing=dict(n=args.n,seed=0,ratio=ratio,epsilon=eps,config=asdict(cfg),num_planes=tree.num_planes(),
                    nodes_per_plane=metadata['nodes_per_plane'],finite=finite,
                    max_abs_difference_from_normal_evaluation=comparison,
                    timings={'walk_near':dict(ms=samples,median_ms=float(np.median(samples)))})
        (dest/'timings.json').write_text(json.dumps(timing,indent=2))
        with jax.profiler.trace(str(dest/'trace')):
            for step in range(args.trace_rounds):
                with jax.profiler.StepTraceAnnotation('tree_walk_near',step_num=step):call()
        subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_uniform_planes.py')),str(dest),
                        '--step-name','tree_walk_near'],stdout=(dest/'summary.log').open('w'),check=True)
        print(f'Finished {i}: {np.median(samples):.3f} ms',flush=True)


if __name__=='__main__':main()
