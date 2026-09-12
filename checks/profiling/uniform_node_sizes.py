"""Measure geometric cell half-diagonals in units of mean particle spacing."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import jax
import numpy as np
from jztree_utils import ics
from jzfmm.config import FMMConfig
from jzfmm.fmm import fast_multipole_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=4_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = FMMConfig()
    part = jax.block_until_ready(ics.uniform_particles(args.n, seed=args.seed))
    tree = jax.block_until_ready(fast_multipole_method.jit(part, cfg, result='tree'))
    spacing = args.n**(-1/3)
    result = dict(n=args.n, seed=args.seed, config=asdict(cfg), spacing=spacing,
                  softening_comparison=3*spacing,
                  radius_definition='Half the cell bounding-box diagonal; not a fitted particle-cloud radius',
                  planes=[])
    for plane in range(tree.num_planes()):
        count = int(tree.num(plane))
        levels = np.asarray(tree.lvl.get(plane, count), dtype=np.int64)
        populations = np.asarray(tree.npart(plane, size=count), dtype=np.int64)
        assert np.all(populations > 0) and populations.sum() == args.n
        # Same per-axis exponent mapping as CUDA lvl_vec<3> (Python floor division).
        quot, rem = np.divmod(levels, 3)
        exponents = quot[:, None] + (rem[:, None] >= np.array([3, 2, 1])[None, :])
        extents = np.exp2(exponents.astype(np.float64))
        radii = .5*np.linalg.norm(extents, axis=1)/spacing
        assert np.isfinite(radii).all()
        unique, inverse, counts = np.unique(levels, return_inverse=True, return_counts=True)
        histogram = []
        for i, level in enumerate(unique):
            mask = inverse == i
            histogram.append(dict(morton_level=int(level), nodes=int(counts[i]),
                                  particles=int(populations[mask].sum()),
                                  radius_over_spacing=float(radii[mask][0]),
                                  side_lengths_over_spacing=(extents[mask][0]/spacing).tolist()))
        hist_sorted = sorted(histogram, key=lambda h:h['radius_over_spacing'])
        cumulative = np.cumsum([h['particles'] for h in hist_sorted])
        particle_median = hist_sorted[int(np.searchsorted(cumulative, args.n/2))]['radius_over_spacing']
        row = dict(plane=plane, nodes=count, mean_particles_per_node=float(populations.mean()),
                   radius_over_spacing=dict(mean=float(radii.mean()),
                       p10=float(np.quantile(radii,.1)), median=float(np.median(radii)),
                       p90=float(np.quantile(radii,.9)), p95=float(np.quantile(radii,.95)),
                       min=float(radii.min()),max=float(radii.max())),
                   particle_weighted_radius_over_spacing=dict(mean=float(np.average(radii,weights=populations)),median=particle_median),
                   fraction_nodes_radius_below_3_spacing=float(np.mean(radii<3)),
                   histogram=histogram)
        result['planes'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='histogram'}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
