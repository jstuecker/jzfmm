# Cell half-diameter relative to mean particle separation

Same default tree as the timing experiments: 4,000,000 independent uniform
float32 positions in [0,1)^3, seed 0; maximum leaf size 32, coarse factor 4.
The tree geometry does not depend on the kernel softening or opening criterion.
Mean particle separation is l = (V/N)^(1/3) = 0.006299605249474368.

Here **half-diameter means half the diagonal of the node's geometric bounding
box**, R = 0.5 sqrt(Lx² + Ly² + Lz²), using the same per-axis cell extents as
the opening criterion. It is not half the side length, a fitted sphere, or the
maximum observed particle distance from the center. The pairwise opening
criterion uses half the norm of the summed extent vectors; that equals R_A+R_B
for cells of the same shape, but need not do so for differently shaped cells.

Every populated node is counted once below. No padded entries are included.

| Plane | Nodes | Mean particles per node | Mean R/l | Median R/l | 10th–90th percentile R/l |
|---|---:|---:|---:|---:|---:|
| Leaves (0) | 177,069 | 22.59 | 2.576 | 2.148 | 2.148–3.038 |
| 1 | 41,741 | 95.83 | 4.049 | 4.296 | 3.720–4.296 |
| 2 | 9,323 | 429.05 | 7.110 | 7.441 | 6.076–7.441 |
| 3 | 2,077 | 1925.85 | 12.052 | 12.151 | 12.151–12.151 |

For the requested planes 0–2, a useful rough summary is **R ≈ 2.6l, 4.0l, 7.1l**.
The discrete median values reflect the binary cell geometry. The mean and
median differ noticeably for leaves because their two dominant cell sizes
have nearly equal numbers of nodes.

At epsilon = 3l, softening is comparable to a leaf radius, smaller than a plane-1
radius, and much smaller than a plane-2 radius. Specifically, epsilon divided
by the mean R is approximately 1.16, 0.74 and 0.42 on planes 0, 1 and 2.
About 51.95% of leaves have R < epsilon; no plane-1 or plane-2 nodes do.
This comparison by itself does not predict opening: the criterion depends on
both cells' sizes, their separation and theta.

## Weighting by particles

For the node containing a randomly selected particle, weight each cell by its
particle population instead of counting nodes equally:

| Plane | Particle-weighted mean R/l | Particle-weighted median R/l |
|---|---:|---:|
| 0 | 2.665 | 3.038 |
| 1 | 4.121 | 4.296 |
| 2 | 7.239 | 7.441 |
| 3 | 12.098 | 12.151 |

## Discrete sizes

### Plane 0

| R/l | Side lengths / l | Nodes | Fraction of nodes |
|---|---|---:|---:|
| 1.860236 | 1.240157, 2.480314, 2.480314 | 30 | 0.017% |
| 2.148015 | 2.480314, 2.480314, 2.480314 | 91,954 | 51.931% |
| 3.037752 | 2.480314, 2.480314, 4.960628 | 85,084 | 48.051% |
| 3.720471 | 2.480314, 4.960628, 4.960628 | 1 | 0.001% |

### Plane 1

| R/l | Side lengths / l | Nodes | Fraction of nodes |
|---|---|---:|---:|
| 3.720471 | 2.480314, 4.960628, 4.960628 | 17,946 | 42.994% |
| 4.296030 | 4.960628, 4.960628, 4.960628 | 23,795 | 57.006% |

### Plane 2

| R/l | Side lengths / l | Nodes | Fraction of nodes |
|---|---|---:|---:|
| 6.075504 | 4.960628, 4.960628, 9.921257 | 2,262 | 24.263% |
| 7.440942 | 4.960628, 9.921257, 9.921257 | 7,061 | 75.737% |

## Method and checks

Build the tree with the existing `fast_multipole_method(..., result='tree')`.
For each plane, extract only `tree.num(plane)` valid Morton levels and decode
the per-axis exponents exactly as CUDA `lvl_vec<3>` does. For binary level k,
q = floor(k/3), m = k mod 3, giving exponents
(q + [m>=3], q + [m>=2], q + [m>=1]). Cell sides are 2 to these powers.
Use float64 on the host to calculate half-diagonals and summary statistics.

The script checks that node populations are positive, sum to 4 million on
every plane, and that all radii are finite. Node counts reproduce the earlier
timing tree. Decoded cell volumes sum to 0.99998856 for leaves and exactly 1.0 for each
coarser plane, consistent with the unit-cube geometry. JSON retains the complete per-level node and particle histograms, so
all weighting choices can be reconstructed.

Reproduce from the fork with the existing CUDA-enabled environment:

```bash
PYTHONPATH="$PWD/src" /home/bender/.venvs/jzfmm/bin/python   checks/profiling/uniform_node_sizes.py --output /tmp/node_sizes.json
```

A CUDA rebuild is not necessary for this tree-only measurement.
