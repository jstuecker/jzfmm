"""Synchronized total timings and CUDA trace for a default uniform FMM."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import time
import jax
import numpy as np
from jztree_utils import ics
from jzfmm.config import FMMConfig
from jzfmm.fmm import fast_multipole_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=4_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--trace-rounds', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert jax.devices()[0].platform == 'gpu'
    cfg = FMMConfig()
    part = jax.block_until_ready(ics.uniform_particles(args.n, seed=args.seed))
    partz, tree = jax.block_until_ready(fast_multipole_method.jit(part, cfg, result='partz_tree'))
    calls = {
        'full': lambda: fast_multipole_method.jit(part, cfg, result='loc'),
        'existing_tree': lambda: fast_multipole_method.jit(partz, cfg, th=tree, result='locz'),
    }
    result = dict(n=args.n, seed=args.seed, config=dataclasses.asdict(cfg),
                  device=str(jax.devices()[0]), num_planes=tree.num_planes(),
                  nodes_per_plane=[int(tree.num(i)) for i in range(tree.num_planes())],
                  versions={p: importlib.metadata.version(p) for p in ['jzfmm','jztree','jax','jaxlib']}, timings={})
    for name, call in calls.items():
        for _ in range(4):
            out = jax.block_until_ready(call())
        samples = []
        for _ in range(args.rounds):
            t = time.perf_counter()
            out = jax.block_until_ready(call())
            samples.append((time.perf_counter()-t)*1000)
        result['timings'][name] = dict(ms=samples, median_ms=float(np.median(samples)), finite=bool(np.isfinite(np.asarray(out.values)).all()))
        print(name, result['timings'][name], flush=True)
    (args.output/'timings.json').write_text(json.dumps(result, indent=2))
    with jax.profiler.trace(str(args.output/'trace')):
        for i in range(args.trace_rounds):
            with jax.profiler.StepTraceAnnotation('full_force', step_num=i):
                jax.block_until_ready(calls['full']())
    print('Trace complete', flush=True)


if __name__ == '__main__':
    main()
