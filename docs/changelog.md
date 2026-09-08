# Changelog

## 1.0.1

- Fixed invalid expansion scaling for tree cells spanning positive and negative coordinates and for tiny or coincident-particle cells. These cases could produce NaNs in FMM results and gradients, including MMD losses used in differentiable simulations. Updating is recommended, especially for differentiable workloads; the fix requires the updated CUDA backend as well as the Python package.
- Added regression tests for forces, gradients, and reversible integration, with a compact subset retained in `--quick` mode.
- Improved the `hello_world.py` dependency hints and refreshed the getting-started guide and example outputs.

## 1.0.0

First public release.
