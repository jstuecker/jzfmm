# Softened opening: uniform 4-million-particle experiment

The requested softened-distance criterion reduces full force-evaluation time
from **188.61 ms to 172.88 ms (8.34% less time;
1.091× speedup)** at the same enlarged kernel softening.
Leaf-leaf work drops by about 21.1 ms, but plane 0 becomes about 3.3 ms more
expensive. Thus the change shifts some work into node interactions instead of
simply eliminating the closest interactions.

## Configuration and criterion

Same 4,000,000 float32 independent uniform particles in [0,1)^3 as the previous
experiment, seed 0, equal masses and total mass 1. Mean separation is defined
as the volume-based spacing l = (V/N)^(1/3), not the mean nearest-neighbor distance.

- l = 0.006299605249474368
- epsilon = 3l = **0.018898815748423106**
- Plummer kernel; p=5; theta=0.8; all remaining FMMConfig defaults unchanged.
- GTX 1070, 8 GiB; driver 580.178.04; Ryzen 5 3600; 16 GiB RAM.
- Tree planes 0–3 contain 177,069 / 41,741 / 9,323 / 2,077 nodes.

For finite cells, the original criterion opens when

    0.25 * ||extent_A + extent_B||^2 >= theta^2 * r^2

The new `OpeningBySoftenedAngle` opens when

    0.25 * ||extent_A + extent_B||^2 >= theta^2 * (r^2 + epsilon^2)

This is exactly the requested substitution sqrt(r² + epsilon²) for r,
implemented with scaled squared quantities to avoid overflow. Epsilon is
included in the numerical scale. The same criterion is instantiated in both
CountInteractionsAndM2L and InsertInteractions. Unbounded nodes still open.
The original criterion and defaults remain available and unchanged. The
criterion's softening is explicitly set equal to the Plummer softening.

This is an experimental convergence approximation, not a force-error guarantee.

## Timings

30 synchronized, unprofiled repetitions per mode, after four warmup calls.
Input generation, transfers and JIT compilation are excluded. Full calls rebuild
sorting, tree and multipoles, and return potential/force in original order.
Existing-tree calls start with sorted particles and a ready tree, but still
rebuild multipoles and execute the complete force evaluation.

| Case | Kernel epsilon | Opening | Full call (ms) | Existing-tree call (ms) |
|---|---:|---|---:|---:|
| Rebuilt original defaults | 0.00100000 | OpeningByAngle | 187.99 | 166.59 |
| Larger-softening control | 0.01889882 | OpeningByAngle | 188.61 | 168.47 |
| Requested experiment | 0.01889882 | OpeningBySoftenedAngle | 172.88 | 151.26 |

The rebuilt original-default result is within 0.1% of the earlier 188.11 ms
measurement using the released binary. The criterion comparison uses the same
rebuilt ffi_fmm extension for both cases. The existing-tree reduction at fixed
epsilon is **10.22% (1.114× speedup)**.

The following component medians come from 10 additional CUDA-profiled full calls.
Node2Node includes the M2L/count kernel, count-array initialization, prefix scans,
list bookkeeping and insertion; local-to-local shifts are in Other GPU work.
Plane 0 is the finest node plane. These are hierarchy planes, not radial shells.

| Component | Original defaults (ms) | Larger epsilon, geometric (ms) | Larger epsilon, softened (ms) | Change at fixed epsilon (ms) |
|---|---:|---:|---:|---:|
| Near field: leaf-leaf | 55.67 | 56.29 | 35.18 | -21.11 |
| Node2Node plane 0 | 51.66 | 52.15 | 55.44 | +3.28 |
| Node2Node plane 1 | 18.99 | 19.18 | 19.41 | +0.24 |
| Node2Node plane 2 | 6.62 | 6.68 | 6.89 | +0.22 |
| Node2Node plane 3 | 15.60 | 15.76 | 15.92 | +0.16 |
| Other GPU work | 41.18 | 41.44 | 41.35 | -0.09 |
| Host / GPU gaps | 0.90 | 0.96 | 1.02 | +0.06 |

- rebuilt-default: profiled full-call median 190.63 ms, 1.41% above its unprofiled median.
- large-eps-geometric: profiled full-call median 192.45 ms, 2.03% above its unprofiled median.
- large-eps-softened: profiled full-call median 175.35 ms, 1.43% above its unprofiled median.

Component medians need not sum exactly to the total median. Do not add the
profiled component medians to an unprofiled residual and treat that as one
measurement. Full-call improvements above use only unprofiled timings.

The leaf-leaf reduction is about 37.5%. Plane-0 M2L/count rises from 47.05 to
50.12 ms: more nearby interactions pass the acceptance test and are handled as
multipole interactions. Small increases in other planes combine changed work,
criterion overhead and run variability; they have not been causally separated.
Both old and new p=5/3D/float32 M2L kernels compiled to 255 registers and 15,424
bytes shared memory, with zero reported register spills on sm_61.

## Sampled force accuracy

128 randomly selected target particles (selection seed 123) were compared with
a float64 direct Plummer sum over all 4 million sources at the same epsilon.
Reference sums remove the self potential. FMM outputs remain float32. These
samples are an accuracy spot check, not a worst-case guarantee.

| Metric | Geometric, larger epsilon | Softened, larger epsilon |
|---|---:|---:|
| Relative force L2 error | 0.029460% | 0.032401% |
| Median relative force error | 0.018549% | 0.018981% |
| 95th percentile relative force error | 0.081853% | 0.087959% |
| Maximum sampled relative force error | 0.161811% | 0.217464% |
| Relative potential L2 error | 0.000263% | 0.000281% |

Relative force L2 is ||F_FMM − F_direct||₂ / ||F_direct||₂ over all sampled vector
components. Per-target quantiles use ||delta F_i|| / ||F_i||. All full outputs
were finite. Two GPU regression tests passed: zero criterion softening gives
bit-identical forces and interaction lists to the original criterion; positive
softening reduces the near-field interaction list on an 8,192-particle case.

## Limits and reproduction

One particle seed, one GPU, unlocked clocks and an active desktop. The cases
were run sequentially; their small timing differences include thermal/clock
variation. JSON and CSV files retain all samples and full configuration.
Raw CUPTI traces and full output arrays are retained locally under
`/home/bender/Work/benchmarks/jzfmm-softened-2026-09-12`, not committed.

Rebuild the CUDA extension with the new opening kind before running these
commands; the released binary does not contain kind 1. See BUILD.md for the
exact local toolchain and build setup. With that extension on PYTHONPATH:

```bash
python checks/profiling/uniform_planes.py --output /tmp/geometric --softening 0.018898815748423106 --save-output
python checks/profiling/uniform_planes.py --output /tmp/softened --softening 0.018898815748423106 --softened-opening --save-output
python checks/profiling/summarize_uniform_planes.py /tmp/geometric
python checks/profiling/summarize_uniform_planes.py /tmp/softened
python checks/profiling/check_uniform_force_samples.py /tmp/geometric /tmp/softened --output /tmp/accuracy.json
python -m pytest checks/tests/test_softened_opening.py -o addopts= -q
```
