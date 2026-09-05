# jz-fmm

**jz-fmm** (*JAX z-order tree, Fast Multipole Method*) is a GPU-native fast
multipole implementation for gravitational N-body simulations. It supports
single- and multi-GPU force evaluation, time integration, and differentiation
through simulations with JAX.

The code builds on [jz-tree](https://github.com/jstuecker/jztree) for tree
construction, traversal, and distributed communication.

## Documentation

The [jz-fmm documentation](https://jstuecker.github.io/jzfmm/) contains the
[installation guide](https://jstuecker.github.io/jzfmm/installation.html),
[getting-started guide](https://jstuecker.github.io/jzfmm/quickstart.html), API
reference, JAX compatibility notes, and performance results.

Installation supports CUDA 12 and CUDA 13. Pre-built wheels are the simplest
option; source builds require a CUDA toolkit containing `nvcc`. See the
installation guide for the appropriate setup and commands for each CUDA
version.
