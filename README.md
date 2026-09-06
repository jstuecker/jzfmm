# jz-fmm

**jz-fmm** (*JAX z-order tree, Fast Multipole Method*) provides fast, GPU-native force calculations and differentiable gravitational N-body simulations in JAX, with a performance-critical CUDA backend. It builds on [jz-tree](https://jstuecker.github.io/jztree/) for tree construction, traversal, and distributed communication.

It includes:

- Fast multipole force evaluation on single and multiple GPUs.
- Time integration and external potential contributions.
- Differentiation through force calculations and simulations with JAX.

**[Documentation](https://jstuecker.github.io/jzfmm/)** · [Installation](https://jstuecker.github.io/jzfmm/installation.html) · [Getting started](https://jstuecker.github.io/jzfmm/quickstart.html) · [Attribution](https://jstuecker.github.io/jzfmm/attribution.html)

## Performance

![Force accuracy versus evaluation time for jz-fmm, GADGET-4, and PKDGRAV3](docs/_static/hernquist_performance_comparison.png)

Comparison of force accuracy and evaluation time for a Hernquist sphere with 40 million particles. All codes were benchmarked on one node with four NVIDIA A100 GPUs and a 32-core CPU; a fairer hardware allocation for the CPU-only GADGET-4 comparison would use approximately 4–8 times as many CPU cores. See the [documentation](https://jstuecker.github.io/jzfmm/) for benchmark details and multi-GPU scaling results.

## Attribution

**jz-fmm** was written by Jens Stücker and is available under the MIT license. If you publish work using **jz-fmm**, please cite the accompanying paper once its citation information is available. See the [attribution page](https://jstuecker.github.io/jzfmm/attribution.html) for funding and acknowledgement details.
